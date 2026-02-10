# assign_af_fluorophores_rcpp.r

#' @title Assign AF Spectrum By Fluorophore Projection Using C++
#'
#' @description
#' Projects the autofluorescence spectral variants into fluorophore unmixed
#' space to determine which best fits each cell (event). A fast approximation for
#' brute force sequential unmixing method in early versions of AutoSpectral.
#' Provides essentially identical results to minimization of fluorophore signal
#' (dist0 method). Substantially faster. Uses OpenMP parallelization in C++.
#'
#' @param raw.data Expression data from raw FCS files. Cells in rows and
#' detectors in columns. Columns should be fluorescent data only and must
#' match the columns in spectra.
#' @param spectra Spectral signatures of fluorophores, normalized between 0
#' and 1, with fluorophores in rows and detectors in columns.
#' @param af.spectra Spectral signatures of autofluorescences, normalized
#' between 0 and 1, with fluorophores in rows and detectors in columns.
#' @param threads Numeric, number of threads to use for parallel processing in
#' C++. Default is `1`, which will use sequential processing.
#'
#'
#' @return Row indices for best-fitting AF spectra (from `af.spectra`)
#'
#' @export

assign.af.fluorophores.rcpp <- function(
    raw.data,
    spectra,
    af.spectra,
    threads = 1
  ) {

  # linear algebra pre-calculations via BLAS
  S <- t( spectra )
  XtX <- tcrossprod( spectra )
  unmixing.matrix <- solve.default( XtX, spectra )

  # how much each AF variant looks like each fluorophore
  v.library <- unmixing.matrix %*% t( af.spectra )

  # calculate the 'residual AF' (the part of AF fluorophores can't explain)
  r.library <- t( af.spectra ) - ( S %*% v.library )

  # predicted AF intensity
  numerator <- raw.data %*% r.library

  # denominator (vector of length af.n)
  denominator <- colSums( r.library^2 )

  # k_matrix (cell.n x af.n): estimated AF intensity per cell/variant
  k.matrix <- sweep( numerator, 2, denominator, "/" )

  # initial unmix (no AF)
  unmixed <- raw.data %*% t( unmixing.matrix )

  # pass the slow part (assignment per cell) to C++
  results <- parallel_af_assign(
    unmixed,
    k.matrix,
    v.library,
    threads = threads
  )

  return( results )
}
