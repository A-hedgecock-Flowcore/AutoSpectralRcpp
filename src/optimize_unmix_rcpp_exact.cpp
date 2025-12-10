// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// Helper: Solve for coefficients directly (S'W S) c = S'W r
// Returns true if solve was successful.
// Writes result into 'coefs'. Uses pre-allocated 'A' and 'b' buffers.
inline bool solve_coefs_fast(arma::vec& coefs,
                             const arma::mat& S,      // P x D (Active spectra)
                             const arma::rowvec& r,   // 1 x D (Raw data)
                             const arma::vec& w,      // D x 1 (Weights)
                             const bool use_weighted,
                             arma::mat& A,            // Buffer: P x P
                             arma::vec& b,            // Buffer: P x 1
                             arma::mat& S_scratch) {  // Buffer: P x D

  if (!use_weighted) {
    // Unweighted OLS: (S S') c = S r'
    // Note: S is (P x D), so we calculate S * S.t()
    A = S * S.t();
    b = S * r.t();
  } else {
    // Weighted OLS: (S W S') c = S W r'
    // We use S_scratch to store (S * diag(W)) to avoid re-allocating
    S_scratch = S;
    for (arma::uword j = 0; j < S.n_cols; ++j) {
      S_scratch.col(j) *= w(j);
    }

    A = S_scratch * S.t();
    b = S_scratch * r.t();
  }

  // Attempt fast solve (Cholesky/LU)
  bool ok = arma::solve(coefs, A, b, arma::solve_opts::fast);

  // Fallback to pinv if singular (rank deficient)
  if (!ok) {
    coefs = arma::pinv(A) * b;
  }

  return true;
}

// [[Rcpp::export]]
arma::mat optimize_unmix_rcpp_exact(const arma::mat& remaining_raw,
                                    const arma::mat& unmixed,
                                    const arma::mat& spectra,
                                    const arma::vec& pos_thresholds,
                                    const arma::uvec& optimize_idx_r,
                                    const std::vector<arma::mat>& variantsList,
                                    const arma::vec& weights,
                                    const bool weighted = false,
                                    const int nthreads = 1) {

  const arma::uword n_cells = remaining_raw.n_rows;
  const arma::uword n_detectors = remaining_raw.n_cols;
  const arma::uword n_fluors = spectra.n_rows;

  arma::mat result = unmixed;

  // Validate weights
  const bool use_weighted = weighted && (weights.n_elem == n_detectors);
  const arma::vec w_vec = use_weighted ? weights : arma::vec();

#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
  const int nthreads_actual = omp_get_max_threads();
#else
  const int nthreads_actual = 1;
#endif

  // --- Pre-allocation Strategy ---
  // We allocate a struct of working buffers for each thread
  struct ThreadBuffers {
    arma::rowvec raw_row;
    arma::rowvec cell_unmixed_row;
    arma::rowvec fitted;
    arma::rowvec resid;
    arma::vec coefs;             // for pos subset
    arma::vec coefs_full;        // for final full solve
    arma::mat cell_spectra;      // P x D (active subset)
    arma::mat final_spectra_pos; // P x D (best-for-pos subset)
    arma::mat final_spectra_full;// n_fluors x D (full spectra with chosen variants)

    // Solver buffers for subset solves
    arma::mat A;                 // P x P (but preallocated larger)
    arma::vec b;                 // P x 1
    arma::mat S_scratch;         // P x D scratch for weighted subset

    // Solver buffers for full solve
    arma::mat S_scratch_full;    // n_fluors x D
    arma::mat A_full;            // n_fluors x n_fluors
    arma::vec b_full;            // n_fluors x 1

    // Logic buffers
    std::vector<arma::uword> pos_vec;
    std::vector<std::pair<double, arma::uword>> order_vec;
  };

  std::vector<ThreadBuffers> thread_buffers(nthreads_actual);

  // Initialize buffers to reasonable max sizes to minimize reallocs
  for(auto& tb : thread_buffers) {
    tb.raw_row.set_size(n_detectors);
    tb.fitted.set_size(n_detectors);
    tb.resid.set_size(n_detectors);
    tb.coefs.set_size(n_fluors);         // over-allocated; we'll use first P entries
    tb.coefs_full.set_size(n_fluors);
    tb.cell_spectra.set_size(n_fluors, n_detectors);      // over-allocated to max P
    tb.final_spectra_pos.set_size(n_fluors, n_detectors); // over-allocated to max P
    tb.final_spectra_full = spectra;                      // n_fluors x n_detectors

    tb.A.set_size(n_fluors, n_fluors);    // reuse top-left PxP region
    tb.b.set_size(n_fluors);
    tb.S_scratch.set_size(n_fluors, n_detectors);

    tb.S_scratch_full.set_size(n_fluors, n_detectors);
    tb.A_full.set_size(n_fluors, n_fluors);
    tb.b_full.set_size(n_fluors);

    tb.pos_vec.reserve(n_fluors);
    tb.order_vec.reserve(n_fluors);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (arma::uword ci = 0; ci < n_cells; ++ci) {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    ThreadBuffers& tb = thread_buffers[tid];

    // Load Data
    tb.raw_row = remaining_raw.row(ci);
    tb.cell_unmixed_row = result.row(ci);

    // 1. Identify positive fluorophores
    tb.pos_vec.clear();
    for (arma::uword f = 0; f < n_fluors; ++f) {
      if (tb.cell_unmixed_row(f) >= pos_thresholds(f)) {
        tb.pos_vec.push_back(f);
      }
    }

    if (tb.pos_vec.empty()) {
      // No positives -> optionally do full solve on original spectra if you prefer,
      // but we keep original behavior: leave result.row(ci) unchanged.
      continue;
    }

    // 2. Setup Active Spectra Matrix (pos subset)
    arma::uword pos_n = tb.pos_vec.size();
    tb.cell_spectra.set_size(pos_n, n_detectors);
    tb.final_spectra_pos.set_size(pos_n, n_detectors);

    // Ensure final_spectra_full is current copy of global spectra
    tb.final_spectra_full = spectra;

    for (arma::uword i = 0; i < pos_n; ++i) {
      tb.cell_spectra.row(i) = spectra.row(tb.pos_vec[i]);
      tb.final_spectra_pos.row(i) = tb.cell_spectra.row(i);
    }

    // 3. Initial Unmix (Solve directly for coefs on the small subset)
    // Use only the top-left PxP region of A and first P entries of b/coefs
    tb.A.set_size(pos_n, pos_n);
    tb.b.set_size(pos_n);
    tb.S_scratch.set_size(pos_n, n_detectors);
    tb.coefs.set_size(pos_n);

    solve_coefs_fast(tb.coefs, tb.cell_spectra, tb.raw_row, w_vec, use_weighted,
                     tb.A, tb.b, tb.S_scratch);

    // 4. Calculate Initial Error (RSS)
    tb.fitted = tb.coefs.t() * tb.cell_spectra;
    tb.resid = tb.raw_row - tb.fitted;
    double best_error = arma::dot(tb.resid, tb.resid);

    // 5. Order fluorophores by intensity (using subset coefs)
    tb.order_vec.clear();
    for (arma::uword i = 0; i < pos_n; ++i) {
      tb.order_vec.emplace_back(tb.coefs(i), i);
    }
    std::sort(tb.order_vec.begin(), tb.order_vec.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // 6. Variant Optimization Loop (subset-level)
    for (const auto& pr : tb.order_vec) {
      arma::uword local_vi = pr.second;              // 0..pos_n-1
      arma::uword f_idx = tb.pos_vec[local_vi];      // actual global fluor index

      // Skip if no variants for this fluorophore
      if (f_idx >= variantsList.size()) continue;
      const arma::mat& fl_variants = variantsList[f_idx];
      if (fl_variants.n_rows == 0) continue;

      // Iterate variants for this fluor
      for (arma::uword v = 0; v < fl_variants.n_rows; ++v) {
        // Swap spectrum for the local position
        tb.cell_spectra.row(local_vi) = fl_variants.row(v);

        // Solve for subset coefs
        tb.A.set_size(pos_n, pos_n);
        tb.b.set_size(pos_n);
        tb.S_scratch.set_size(pos_n, n_detectors);
        tb.coefs.set_size(pos_n);

        solve_coefs_fast(tb.coefs, tb.cell_spectra, tb.raw_row, w_vec, use_weighted,
                         tb.A, tb.b, tb.S_scratch);

        // Compute Error for this candidate
        tb.fitted = tb.coefs.t() * tb.cell_spectra;
        tb.resid = tb.raw_row - tb.fitted;
        double error_curr = arma::dot(tb.resid, tb.resid);

        if (error_curr < best_error) {
          best_error = error_curr;
          // Update the "Best" state: place variant into final spect full matrix
          tb.final_spectra_pos.row(local_vi) = fl_variants.row(v);
          tb.final_spectra_full.row(f_idx) = fl_variants.row(v); // update global row
          // keep tb.cell_spectra row as-is (holds best variant)
        } else {
          // Revert the change in the local subset matrix
          tb.cell_spectra.row(local_vi) = tb.final_spectra_pos.row(local_vi);
        }
      } // variants loop

      // Sync cell_spectra with the known best for this position
      tb.cell_spectra.row(local_vi) = tb.final_spectra_pos.row(local_vi);

      // Re-solve to get updated coefficients for ordering/next iterations
      tb.A.set_size(pos_n, pos_n);
      tb.b.set_size(pos_n);
      tb.S_scratch.set_size(pos_n, n_detectors);
      tb.coefs.set_size(pos_n);

      solve_coefs_fast(tb.coefs, tb.cell_spectra, tb.raw_row, w_vec, use_weighted,
                       tb.A, tb.b, tb.S_scratch);
    } // end order_vec loop

    // ------------------------------------------------------------------------
    // FINAL FULL SOLVE (single dense solve for all fluorophores)
    // Use tb.final_spectra_full (n_fluors x n_detectors) which has variant
    // replacements applied for the positive fluorophores chosen above.
    // ------------------------------------------------------------------------

    // Prepare full-solve buffers
    tb.S_scratch_full = tb.final_spectra_full; // copy full spectra
    if (use_weighted) {
      for (arma::uword j = 0; j < n_detectors; ++j) {
        tb.S_scratch_full.col(j) *= w_vec(j);
      }
    }

    tb.A_full = tb.S_scratch_full * tb.final_spectra_full.t(); // n_fluors x n_fluors
    tb.b_full = tb.S_scratch_full * tb.raw_row.t();            // n_fluors x 1

    tb.coefs_full.set_size(n_fluors);
    bool ok_full = arma::solve(tb.coefs_full, tb.A_full, tb.b_full, arma::solve_opts::fast);
    if (!ok_full) {
      tb.coefs_full = arma::pinv(tb.A_full) * tb.b_full;
    }

    // Write full coefficients back into result
    result.row(ci) = tb.coefs_full.t();
  } // end cells loop

  return result;
}
