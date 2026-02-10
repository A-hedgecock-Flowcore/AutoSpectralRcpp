#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace arma;

// [[Rcpp::export]]
Rcpp::List parallel_unmix_af(const arma::mat& raw_data,
                             const arma::mat& af_spectra,
                             const arma::mat& fluor_spectra,
                             const arma::vec& af_assignments,
                             int n_threads = 4) {

  int n_cells = raw_data.n_rows;
  int n_channels = raw_data.n_cols;
  int n_fluors = fluor_spectra.n_rows;
  arma::vec unique_af = arma::unique(af_assignments);
  int n_groups = unique_af.n_elem;

  arma::mat unmixed(n_cells, n_fluors + 1, fill::zeros);
  arma::mat fitted_af(n_cells, n_channels, fill::zeros);

  // Set number of threads
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif

#pragma omp parallel
{
  // Crucial: Each thread needs its OWN copy of the design matrix
  arma::mat local_combined(n_fluors + 1, n_channels);
  local_combined.rows(0, n_fluors - 1) = fluor_spectra;

#pragma omp for
  for (int i = 0; i < n_groups; ++i) {
    double current_af_val = unique_af(i);
    int af_row_idx = (int)current_af_val - 1;

    local_combined.row(n_fluors) = af_spectra.row(af_row_idx);

    arma::uvec cell_idx = find(af_assignments == current_af_val);

    if (cell_idx.n_elem > 0) {
      arma::mat Y = raw_data.rows(cell_idx);

      // Use fast solve
      arma::mat B = arma::solve(local_combined.t(), Y.t(), arma::solve_opts::fast);

      // Critical: writing to unmixed and fitted_af is safe because cell_idx are unique
      unmixed.rows(cell_idx) = B.t();

      arma::vec af_coeffs = B.row(n_fluors).t();
      fitted_af.rows(cell_idx) = af_coeffs * af_spectra.row(af_row_idx);
    }
  }
}

return Rcpp::List::create(Rcpp::_["unmixed"] = unmixed, Rcpp::_["fitted.af"] = fitted_af);
}
