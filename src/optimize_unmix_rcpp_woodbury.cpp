// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// helper: compute unmixing matrix (OLS or WLS)
inline arma::mat compute_unmixing_matrix_general(const arma::mat& spectra,
                                                 const arma::vec& weights,
                                                 bool weighted) {
  arma::mat M = spectra.t(); // D x F
  arma::uword D = M.n_rows;

  if (!weighted || weights.n_elem == 0) {
    arma::mat A = M.t() * M;   // F x F
    arma::mat B = M.t();       // F x D
    arma::mat X;
    bool ok = arma::solve(X, A, B);
    if (!ok) X = arma::pinv(A) * B;
    return X;                  // F x D
  } else {
    arma::mat MtW = M.t();
    for (arma::uword j = 0; j < D; ++j)
      MtW.col(j) *= weights(j);

    arma::mat A = MtW * M;     // F x F
    arma::mat B = MtW;         // F x D
    arma::mat X;
    bool ok = arma::solve(X, A, B);
    if (!ok) X = arma::pinv(A) * B;
    return X;
  }
}

// helper: invert a 2x2 matrix safely
inline bool inv2x2(const arma::mat& M2, arma::mat& out) {
  if (M2.n_rows != 2 || M2.n_cols != 2) return false;
  double a = M2(0,0), b = M2(0,1), c = M2(1,0), d = M2(1,1);
  double det = a*d - b*c;
  if (std::abs(det) < 1e-12) return false;

  out.set_size(2,2);
  out(0,0) =  d/det;
  out(0,1) = -b/det;
  out(1,0) = -c/det;
  out(1,1) =  a/det;
  return true;
}

// [[Rcpp::export]]
arma::mat optimize_unmix_rcpp_woodbury(const arma::mat& remaining_raw,
                                       const arma::mat& unmixed,
                                       const arma::mat& spectra,
                                       const arma::vec& pos_thresholds,
                                       const arma::uvec& optimize_idx_r,
                                       const std::vector<arma::mat>& variantsList,
                                       const arma::vec& weights,
                                       const bool weighted = false,
                                       const int nthreads = 1,
                                       const double singular_tol = 1e-8) {

  const arma::uword n_cells     = remaining_raw.n_rows;
  const arma::uword n_detectors = remaining_raw.n_cols;
  const arma::uword n_fluors    = spectra.n_rows;

  arma::mat result = unmixed;

  // handle weights
  arma::vec w = (weights.n_elem == 0) ? arma::ones<arma::vec>(n_detectors) : weights;
  if (w.n_elem != n_detectors)
    Rcpp::stop("weights length mismatch");

  const bool do_weighted_final = weighted && (w.n_elem == n_detectors);

#ifdef _OPENMP
  omp_set_num_threads(nthreads);
  int nthreads_actual = omp_get_max_threads();
#else
  int nthreads_actual = 1;
#endif

  // per-thread allocations
  std::vector<arma::rowvec> tmp_raw_row(nthreads_actual, arma::rowvec(n_detectors));
  std::vector<arma::rowvec> tmp_cell_unmixed(nthreads_actual, arma::rowvec(n_fluors));
  std::vector<arma::rowvec> tmp_fitted(nthreads_actual, arma::rowvec(n_detectors));
  std::vector<arma::rowvec> tmp_resid(nthreads_actual, arma::rowvec(n_detectors));
  std::vector<arma::rowvec> tmp_resid_candidate(nthreads_actual, arma::rowvec(n_detectors));
  std::vector<arma::mat>    tmp_spectra_final(nthreads_actual, arma::mat(spectra));
  std::vector<arma::mat>    tmp_Mpos(nthreads_actual);
  std::vector<arma::mat>    tmp_Ainv(nthreads_actual);
  std::vector<arma::mat>    tmp_Bpos(nthreads_actual);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (arma::uword ci = 0; ci < n_cells; ++ci) {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif

    arma::rowvec& raw_row          = tmp_raw_row[tid];
    arma::rowvec& cell_unmixed_row = tmp_cell_unmixed[tid];
    arma::rowvec& fitted           = tmp_fitted[tid];
    arma::rowvec& resid            = tmp_resid[tid];
    arma::rowvec& resid_candidate  = tmp_resid_candidate[tid];
    arma::mat&    spectra_final    = tmp_spectra_final[tid];

    raw_row          = remaining_raw.row(ci);
    cell_unmixed_row = result.row(ci);

    //
    // Step 1: identify positive fluorophores
    //
    std::vector<arma::uword> pos_vec;
    pos_vec.reserve(n_fluors);

    for (arma::uword f = 0; f < n_fluors; ++f) {
      if (cell_unmixed_row(f) >= pos_thresholds(f))
        pos_vec.push_back(f);
    }

    if (pos_vec.empty())
      continue;

    arma::uvec pos_idx(pos_vec);
    arma::uword pos_n = pos_idx.n_elem;

    //
    // Step 2: extract spectra subset
    //
    arma::mat cell_spectra(pos_n, n_detectors);
    for (arma::uword i = 0; i < pos_n; ++i)
      cell_spectra.row(i) = spectra.row(pos_idx(i));

    arma::mat& Mpos = tmp_Mpos[tid];
    Mpos = cell_spectra.t();                        // D x pos_n
    arma::mat A = Mpos.t() * Mpos;                  // pos_n x pos_n

    arma::mat Ainv;
    if (!arma::inv(Ainv, A))
      Ainv = arma::pinv(A);

    tmp_Ainv[tid] = Ainv;

    arma::mat Bpos = Ainv * Mpos.t();               // pos_n x D
    tmp_Bpos[tid] = Bpos;

    //
    // Step 3: initial fit & residual
    //
    arma::rowvec unmixed_pos = raw_row * Bpos.t();
    fitted = unmixed_pos * Mpos.t();
    resid  = raw_row - fitted;

    double error_final = arma::accu(arma::abs(resid));

    spectra_final = spectra;

    //
    // Step 4: try variant substitutions (Woodbury)
    //
    std::vector<std::pair<double, arma::uword>> order_vec;
    order_vec.reserve(pos_n);

    for (arma::uword i = 0; i < pos_n; ++i)
      order_vec.emplace_back(unmixed_pos(i), i);

    std::sort(order_vec.begin(), order_vec.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& pr : order_vec) {
      arma::uword local_vi = pr.second;
      arma::uword f_idx    = pos_idx(local_vi);

      if (f_idx >= variantsList.size())
        continue;

      const arma::mat& fl_variants = variantsList[f_idx];
      arma::uword n_var = fl_variants.n_rows;
      if (n_var == 0)
        continue;

      arma::colvec m_old = Mpos.col(local_vi);
      if (unmixed_pos(local_vi) <= 0)
        continue;

      for (arma::uword v = 0; v < n_var; ++v) {
        arma::colvec m_new = fl_variants.row(v).t();
        arma::colvec d = m_new - m_old;
        arma::colvec a = Mpos.t() * d;
        double dd = arma::dot(d, d);

        arma::mat U(pos_n, 2), V(pos_n, 2);
        U.col(0) = a;
        U.col(1).zeros();
        U(local_vi, 1) = 1.0;

        V.col(0).zeros();
        V(local_vi, 0) = 1.0;
        V.col(1) = a;
        V(local_vi, 1) += dd;

        arma::mat AinvU = Ainv * U;
        arma::mat S = arma::eye(2,2) + V.t() * AinvU;

        arma::mat S_inv;
        bool Sok = inv2x2(S, S_inv);

        arma::mat Ainv_candidate;

        if (Sok) {
          arma::mat VtAinv = V.t() * Ainv;
          Ainv_candidate = Ainv - AinvU * (S_inv * VtAinv);
        } else {
          arma::mat Mpos_candidate = Mpos;
          Mpos_candidate.col(local_vi) = m_new;
          arma::mat A_candidate = Mpos_candidate.t() * Mpos_candidate;
          if (!arma::inv(Ainv_candidate, A_candidate))
            Ainv_candidate = arma::pinv(A_candidate);
        }

        arma::mat Mpos_prime = Mpos;
        Mpos_prime.col(local_vi) = m_new;

        arma::mat Bpos_prime = Ainv_candidate * Mpos_prime.t();

        arma::rowvec unmixed_pos_candidate = raw_row * Bpos_prime.t();
        arma::rowvec fitted_cand = unmixed_pos_candidate * Mpos_prime.t();
        resid_candidate = raw_row - fitted_cand;

        double error_curr = arma::accu(arma::abs(resid_candidate));

        if (error_curr < error_final) {
          error_final = error_curr;
          resid = resid_candidate;
          spectra_final.row(f_idx) = fl_variants.row(v);
          Mpos = Mpos_prime;
          Ainv = Ainv_candidate;
          Bpos = Bpos_prime;
          unmixed_pos = unmixed_pos_candidate;
        }
      }
    }

    //
    // Step 5: final unmix with updated spectra
    //
    arma::mat unmix_full = compute_unmixing_matrix_general(
      spectra_final, w, do_weighted_final
    );

    result.row(ci) = raw_row * unmix_full.t();
  }

  return result;
}
