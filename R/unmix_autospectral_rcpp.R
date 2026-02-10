# unmix_autospectral_rcpp.r

#' @title Unmix AutoSpectral Rcpp
#'
#' @description
#' Unmix using the AutoSpectral method to extract autofluorescence and optimize
#' fluorophore signatures at the single cell level.
#'
#' @importFrom lifecycle deprecate_warn
#' @importFrom parallelly availableCores
#' @importFrom AutoSpectral assign.af.residuals
#'
#' @param raw.data Expression data from raw fcs files. Cells in rows and
#' detectors in columns. Columns should be fluorescent data only and must
#' match the columns in spectra.
#' @param spectra Spectral signatures of fluorophores, normalized between 0
#' and 1, with fluorophores in rows and detectors in columns.
#' @param af.spectra Spectral signatures of autofluorescences, normalized
#' between 0 and 1, with fluorophores in rows and detectors in columns. Prepare
#' using `get.af.spectra`.
#' @param asp The AutoSpectral parameter list.
#' @param spectra.variants Named list (names are fluorophores) carrying matrices
#' of spectral signature variations for each fluorophore. Prepare using
#' `get.spectral.variants`. Default is `NULL`.
#' @param use.dist0 Logical, controls whether the selection of the optimal AF
#' signature for each cell is determined by which unmixing brings the fluorophore
#' signals closest to 0 (`use.dist0` = `TRUE`) or by which unmixing minimizes the
#' per-cell residual (`use.dist0` = `FALSE`). Default is `TRUE`.
#' @param verbose Logical, whether to send messages to the console.
#' Default is `TRUE`.
#' @param speed Selector for the precision-speed trade-off in AutoSpectral per-cell
#' fluorophore optimization. Options are `slow`, `medium` and `fast`. From v1.0.0,
#' this controls the number of variants tested per cell (and per fluorophore).
#' More variants takes longer, but gives better resolution in the unmixed data.
#' When `speed = fast`, as single variant will be tested; for `medium`, three
#' will be tested and for `slow`, 10 variants will be tested. From AutoSpectral
#' v1.0.0, `slow` is the default.
#' @param parallel Logical, default is `TRUE`. Always use parallel processing in
#' the C++ version for faster processing.
#' @param threads Numeric, default is `NULL`, in which case `asp$worker.process.n`
#' will be used. `asp$worker.process.n` is set by default to be one less than the
#' available cores on the machine. Multi-threading is only used if `parallel` is
#' `TRUE`.
#' @param k Number of variants (and autofluorescence spectra) to test per cell.
#' Allows explicit control over the number used, as opposed to `speed`, which
#' selects from pre-defined choices. Providing a numeric value to `k` will
#' override `speed`, allowing up to `k` (or the max available) variants to be
#' tested. The default is `NULL`, in which case `k` will be ignored.
#' @param ... Ignored. Previously used for deprecated arguments such as
#' `calculate.error`.
#'
#' @return Unmixed data with cells in rows and fluorophores in columns.
#'
#' @export

unmix.autospectral.rcpp <- function(
    raw.data,
    spectra,
    af.spectra,
    asp,
    spectra.variants = NULL,
    use.dist0 = TRUE,
    verbose = TRUE,
    speed = c("slow", "medium", "fast"),
    parallel = TRUE,
    threads = NULL,
    k = NULL,
    ...
) {

  # warn regarding deprecated arguments
  dots <- list( ... )

  if ( !is.null( dots$calculate.error ) ) {
    lifecycle::deprecate_warn(
      "0.9.1",
      "unmix.autospectral(calculate.error)",
      details = "no longer used"
    )
  }


  #############################################
  ### Autofluorescence Optimization Section ###
  #############################################

  # check for AF in spectra, remove if present
  if ( "AF" %in% rownames( spectra ) )
    spectra <- spectra[ rownames( spectra ) != "AF", , drop = FALSE ]

  if ( is.null( af.spectra ) )
    stop( "Multiple AF spectra must be provided.",
          call. = FALSE )
  if ( nrow( af.spectra ) < 2 )
    stop( "Multiple AF spectra must be provided.",
          call. = FALSE )

  if ( verbose ) message( "Determining initial AF assignments for each cell" )

  # check for data/spectra column matching
  raw.data.cols <- colnames( raw.data )
  spectra.cols <- colnames( spectra )

  if ( !identical( raw.data.cols, spectra.cols ) ) {

    # ensure both actually have the same columns before reordering
    if ( all( spectra.cols %in% raw.data.cols ) &&
         length( spectra.cols ) == length( raw.data.cols ) ) {
      # reorder raw.data to match the order of spectra
      raw.data <- raw.data[, spectra.cols]
      message( "Columns reordered to match spectra." )
    } else {
      stop( "Column names in spectra and raw.data do not match perfectly;
           cannot reorder by name alone." )
    }
  }

  # set up for per-cell AF extraction
  fluorophores <- rownames( spectra )
  fluors.af <- c( fluorophores, "AF" )

  # set number of threads to use
  if ( parallel ) {
    if ( is.null( threads ) ) threads <- asp$worker.process.n
    if ( threads == 0 ) threads <- parallelly::availableCores()
  } else {
    threads <- 1
  }

  # score the AF spectra per cell to determine the initial best "AF Index"
  if ( use.dist0 ) {
    # use minimization of total fluorophore signal (worst case scenario) as the metric
    af.assignments <- assign.af.fluorophores.rcpp(
      raw.data,
      spectra,
      af.spectra,
      threads
    )
  } else {
    # use minimization of/alignment to residuals as the metric
    af.assignments <- AutoSpectral::assign.af.residuals(
      raw.data,
      spectra,
      af.spectra
    )
  }

  if ( verbose ) message( "Unmixing autofluorescence" )

  # unmix using C++ with parallelization
  result <- parallel_unmix_af(
    raw_data = raw.data,
    af_spectra = af.spectra,
    fluor_spectra = spectra,
    af_assignments = af.assignments,
    n_threads = threads
  )

  # add AF assignments
  unmixed <- cbind( result$unmixed, af.assignments )
  colnames( unmixed ) <- c( fluors.af, "AF Index" )

  # if we don't have spectral variants and aren't refining AF, stop here
  if ( is.null( spectra.variants ) ) return( unmixed )


  #############################################
  ##### Per-Cell Fluorophore Optimization #####
  #############################################

  # variable naming
  fluorophore.n <- nrow( spectra )
  af.idx.in.spectra <- fluorophore.n + 1
  detector.n <- ncol( spectra )
  combined.spectra <- matrix(
    NA_real_,
    nrow = af.idx.in.spectra,
    ncol = detector.n
  )
  colnames( combined.spectra ) <- colnames( spectra )
  rownames( combined.spectra ) <- fluors.af
  combined.spectra[ 1:fluorophore.n, ] <- spectra

  # set positivity thresholds vector
  pos.thresholds <- rep( Inf, fluorophore.n )
  names( pos.thresholds ) <- fluorophores
  # fill with data
  pos.thresholds[ names( spectra.variants$thresholds ) ] <- spectra.variants$thresholds

  # unpack spectral variants
  variants <- spectra.variants$variants
  delta.list <- spectra.variants$delta.list
  delta.norms <- spectra.variants$delta.norms

  if ( is.null( pos.thresholds ) )
    stop( "Check that spectral variants have been calculated using
          `get.spectra.variants()`",
          call. = FALSE )
  if ( is.null( variants ) )
    stop( "Multiple fluorophore spectral variants must be provided.,
        call. = FALSE" )
  if ( !( length( variants ) > 1 ) )
    stop( "Multiple fluorophore spectral variants must be provided.",
          call. = FALSE )

  # use AF-subtracted raw data as input for fluorophore optimization
  remaining.raw <- raw.data - result$fitted.af

  # if delta.list and delta.norms are not provided by AutoSpectral (<v1.0.0), calculate
  # this can be done in a single lapply call
  if ( is.null( delta.list ) ) {
    warning(
      paste(
        "For best results, re-calculate `spectra.variants` using AutoSpectral",
        "version 1.0.0 or greater. See `?get.spectra.variants`."
      ),
      call. = FALSE
    )
    # calculate deltas for each fluorophore's variants
    delta.list <- lapply( optimize.fluors, function( fl ) {
      variants[[ fl ]] - matrix(
        spectra[ fl, ],
        nrow = nrow( variants[[ fl ]] ),
        ncol = detector.n,
        byrow = TRUE
      )
    } )
    names( delta.list ) <- optimize.fluors

    # precompute delta norms
    delta.norms <- lapply( delta.list, function( d ) {
      sqrt( rowSums( d^2 ) )
    } )
  }

  # set number of variants to test (`speed`)
  if ( speed == "slow" ) {
    k <- 10L
  } else if ( speed == "medium" ) {
    k <- 3L
  } else if ( speed == "fast" ) {
    k <- 1L
  } else {
    k <- 10L
    warning(
      "Unrecognized input to `speed`. Defaulting to `slow` method.",
      call. = FALSE
    )
  }

  # pre-calculate indices rather than using names
  fluorophores <- which( rownames( combined.spectra ) %in% fluorophores )

  # restrict optimization to fluors present in names( variants )
  optimize.fluors <- fluorophores[ fluorophores %in% names( variants ) ]

  # check inputs for C++
  optimize.fluors <- sanitize.optimization.inputs(
    spectra,
    optimize.fluors,
    variants,
    delta.norms
  )

  # if no match, return unmixed with warning
  if ( !( length( optimize.fluors ) > 0 ) ) {
    warning(
      "No matching fluorophores between supplied spectra and spectral variants.
      No spectral optimization performed.",
      call. = FALSE
    )
    return( unmixed )
  }

  if ( verbose ) message( "Optimizing fluorophore unmixing cell-by-cell" )

  # call per-cell unmixing function
  unmixed[ , fluorophores ] <- optimize_unmix(
    raw_data = remaining.raw,
    unmixed_init = unmixed[ , fluorophores, drop = FALSE ],
    base_spectra = spectra,
    pos_thresholds = pos.thresholds,
    fluor_names = fluorophores,
    optimize_fluors = optimize.fluors,
    all_fluorophores = fluorophores,
    variants = variants,
    delta_list = delta.list,
    delta_norms = delta.norms,
    k = k,
    nthreads = threads
  )

  return( unmixed )

}

