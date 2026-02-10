#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;

// [[Rcpp::export]]
arma::mat optimize_unmix(const arma::mat& raw_data,
                         const arma::mat& unmixed_init,
                         const arma::mat& base_spectra,
                         const arma::vec& pos_thresholds,
                         CharacterVector fluor_names,
                         CharacterVector optimize_fluors,
                         CharacterVector all_fluorophores,
                         List variants,
                         List delta_list,
                         List delta_norms,
                         int k = 10,
                         int nthreads = 1) {

  const arma::uword n_cells = raw_data.n_rows;
  const arma::uword n_all_fluors = base_spectra.n_rows;
  const size_t n_variants_lists = (size_t)variants.size();

  std::vector<std::string> cpp_names = as<std::vector<std::string>>(fluor_names);
  std::vector<std::string> cpp_opt_names = as<std::vector<std::string>>(optimize_fluors);
  std::vector<std::string> v_names = as<std::vector<std::string>>(variants.names());

  std::vector<arma::mat> v_mats(n_variants_lists);
  std::vector<arma::mat> d_mats(n_variants_lists);
  std::vector<arma::vec> dn_vecs(n_variants_lists);
  std::vector<int> var_to_master(n_variants_lists);
  std::vector<bool> should_opt(n_variants_lists, false);

  for (size_t f = 0; f < n_variants_lists; ++f) {
    v_mats[f] = as<arma::mat>(variants[f]);
    d_mats[f] = as<arma::mat>(delta_list[f]);
    dn_vecs[f] = as<arma::vec>(delta_norms[f]);

    // Safeguard: If the norm vector is empty or the max norm is 0 (all zeros), do not optimize
    if (dn_vecs[f].is_empty() || arma::max(dn_vecs[f]) <= 1e-12) {
      should_opt[f] = false;
    } else {
      // Check if this fluor is in the optimization subset
      for (const std::string& opt_name : cpp_opt_names) {
        if (opt_name == v_names[f]) {
          should_opt[f] = true;
          break;
        }
      }
    }

    for (size_t n = 0; n < cpp_names.size(); ++n) {
      if (cpp_names[n] == v_names[f]) {
        var_to_master[f] = (int)n;
        break;
      }
    }
  }

  std::vector<int> exit_indices;
  for (int i = 0; i < all_fluorophores.size(); ++i) {
    std::string target = as<std::string>(all_fluorophores[i]);
    for (size_t n = 0; n < cpp_names.size(); ++n) {
      if (cpp_names[n] == target) {
        exit_indices.push_back((int)n);
        break;
      }
    }
  }

  arma::mat final_output(n_cells, n_all_fluors);

#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif

#pragma omp parallel for schedule(dynamic, 256)
  for (arma::uword i = 0; i < n_cells; ++i) {
    arma::rowvec cell_raw = raw_data.row(i);
    arma::rowvec cell_unmixed = unmixed_init.row(i);
    arma::mat cell_spectra_final = base_spectra;

    bool any_pos = false;
    for (int idx : exit_indices) {
      if (cell_unmixed[idx] >= pos_thresholds[idx]) {
        any_pos = true;
        break;
      }
    }

    if (!any_pos) {
      final_output.row(i) = cell_unmixed;
      continue;
    }

    arma::uvec pos_idx = arma::find(cell_unmixed.t() >= pos_thresholds);
    arma::mat spectra_curr = cell_spectra_final.rows(pos_idx);
    arma::rowvec unmixed_curr = arma::solve(spectra_curr.t(), cell_raw.t(), arma::solve_opts::fast).t();
    arma::rowvec resid = cell_raw - (unmixed_curr * spectra_curr);
    double error_final = arma::sum(arma::abs(resid));

    // Square the residual norm once per cell to speed up scoring
    double resid_norm = std::sqrt(arma::dot(resid, resid));

    for (size_t f = 0; f < n_variants_lists; ++f) {
      // Combined safeguard check
      if (!should_opt[f]) continue;

      int master_idx = var_to_master[f];
      int row_in_curr = -1;
      for(arma::uword p = 0; p < pos_idx.n_elem; ++p) {
        if((int)pos_idx[p] == master_idx) {
          row_in_curr = (int)p;
          break;
        }
      }

      if (row_in_curr != -1 && resid_norm > 1e-12) {
        const arma::mat& fl_vars = v_mats[f];
        const arma::mat& d_fl = d_mats[f];
        const arma::vec& dn = dn_vecs[f];

        arma::vec scores = (d_fl * resid.t()) * unmixed_curr[row_in_curr];
        // Divide by the pre-calculated residual norm and the variant's delta norm
        scores /= (dn * resid_norm);

        arma::uvec sorted_idx = arma::sort_index(scores, "descend");
        int k_eff = std::min(k, (int)sorted_idx.n_elem);

        for (int v_i = 0; v_i < k_eff; ++v_i) {
          int var_idx = (int)sorted_idx[v_i];
          arma::rowvec original_row = spectra_curr.row(row_in_curr);
          spectra_curr.row(row_in_curr) = fl_vars.row(var_idx);

          arma::rowvec t_unmix = arma::solve(spectra_curr.t(), cell_raw.t(), arma::solve_opts::fast).t();
          arma::rowvec t_resid = cell_raw - (t_unmix * spectra_curr);
          double t_err = arma::sum(arma::abs(t_resid));

          if (t_err < error_final) {
            error_final = t_err;
            unmixed_curr = t_unmix;
            resid = t_resid;
            // Update residual norm for the next variant score in this same cell
            resid_norm = std::sqrt(arma::dot(resid, resid));
            cell_spectra_final.row(master_idx) = spectra_curr.row(row_in_curr);
          } else {
            spectra_curr.row(row_in_curr) = original_row;
          }
        }
      }
    }
    final_output.row(i) = arma::solve(cell_spectra_final.t(), cell_raw.t(), arma::solve_opts::fast).t();
  }

  return final_output;
}
