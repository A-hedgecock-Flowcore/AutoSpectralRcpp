# AutoSpectralRcpp 0.2.0 (2025-12-07)

## New features
- Added fast Poisson–IRLS unmixing with incremental updates.
- Added OpenMP support with optimized C++ kernels.
- Implemented new SSM calculation pipeline.

## Improvements
- Faster Poisson–IRLS unmixing with fast QR decomposition
- Better handling of convergence with step halving, deviance monitoring
- Allow early exit if convergence reached
- unmix.wls updated to match AutoSpectral, ensuring non-negative weighting and
a more numerically stable solve.
- Hopefully faster compiler flags.


## Bug fixes
- Initial estimates for IRLS are no longer clamped to non-negative values
- Indentation error in optimize_unmix_rcpp_woodbury

---
