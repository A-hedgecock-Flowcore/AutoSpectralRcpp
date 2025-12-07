#' Poisson IRLS Parallel
#'
#' @export
#' @useDynLib AutoSpectralRcpp, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @param raw_data matrix of cells x detectors
#' @param spectra matrix of fluorophores x detectors
#' @param beta_init initial unmixed matrix of cells x fluorophores
#' @param maxit maximum number of IRLS iterations
#' @param tol convergence tolerance
#' @param n_threads number of threads to use (OpenMP)
#' @param divergence_threshold maximum allowed divergence before fallback
#' @param max_halving_steps maximum number of halving steps to use to approach
#' convergence
poisson_irls_rcpp_parallel <- function(raw_data,
                                       spectra,
                                       beta_init,
                                       maxit = 25L,
                                       tol = 1e-6,
                                       n_threads = 1L,
                                       divergence_threshold = 1e4,
                                       max_halving_steps = 20) {
  .Call(`_AutoSpectralRcpp_poisson_irls_rcpp_parallel`, raw_data, spectra,
        beta_init, maxit, tol, n_threads, divergence_threshold, max_halving_steps)
}
