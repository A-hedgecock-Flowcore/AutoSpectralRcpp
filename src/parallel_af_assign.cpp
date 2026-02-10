#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace arma;

// [[Rcpp::export]]
Rcpp::IntegerVector parallel_af_assign(const arma::mat& unmixed,
                                       const arma::mat& k_matrix,
                                       const arma::mat& v_library,
                                       int threads = 1) {
  int n_cells = unmixed.n_rows;
  int n_af = k_matrix.n_cols;

  Rcpp::IntegerVector best_indices(n_cells);

#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_cells; ++i) {
    double min_error = 1e30;
    int best_idx = 0;

    // We treat the unmixed row as a vector for faster access
    rowvec cell_unmixed = unmixed.row(i);

    for (int j = 0; j < n_af; ++j) {
      // Calculate L1 error: sum(|unmixed_row - k * v_col|)
      // Armadillo's vectorized approach is often faster than a manual 'f' loop
      double total_abs_error = accu(abs(cell_unmixed - k_matrix(i, j) * v_library.col(j).t()));

      if (total_abs_error < min_error) {
        min_error = total_abs_error;
        best_idx = j + 1; // R-compatible indexing
      }
    }
    best_indices[i] = best_idx;
  }

  return best_indices;
}
