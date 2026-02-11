#include <RcppArmadillo.h>
#include <algorithm>
#include <vector>
#include <string>

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

  // Extract data for thread safety and speed
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

    // Default to false for every fluorophore in the variants list
    should_opt[f] = false;

    // First check: Is it even in the user's list of fluors to optimize?
    bool is_requested = false;
    for (const std::string& opt_name : cpp_opt_names) {
      if (opt_name == v_names[f]) {
        is_requested = true;
        break;
      }
    }

    // Second check: If requested, does it actually have valid variants (non-zero norms)?
    if (is_requested) {
      bool has_variants = !dn_vecs[f].is_empty() && (arma::max(dn_vecs[f]) > 1e-12);
      if (has_variants) {
        should_opt[f] = true;
      }
    }

    // Map to master index (independent of optimization status)
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

  // --- parallel optimization loop ---
#pragma omp parallel for schedule(dynamic, 256)
  for (arma::uword i = 0; i < n_cells; ++i) {
    // Thread-local data
    arma::rowvec cell_raw = raw_data.row(i);
    arma::rowvec cell_unmixed = unmixed_init.row(i);
    arma::mat cell_spectra_final = base_spectra;

    // Early Exit Check
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

    // Initial Unmix for present fluors
    arma::uvec pos_idx = arma::find(cell_unmixed.t() >= pos_thresholds);
    arma::mat spectra_curr = cell_spectra_final.rows(pos_idx);
    arma::rowvec unmixed_curr = arma::solve(spectra_curr.t(), cell_raw.t(), arma::solve_opts::fast).t();
    arma::rowvec resid = cell_raw - (unmixed_curr * spectra_curr);
    double error_final = arma::sum(arma::abs(resid));
    double resid_norm = std::sqrt(arma::dot(resid, resid));

    // Sorting: Rank fluors by current abundance
    std::vector<std::pair<double, size_t>> fluor_order;
    for (size_t f = 0; f < n_variants_lists; ++f) {
      if (!should_opt[f]) continue;
      int master_idx = var_to_master[f];
      for(arma::uword p = 0; p < pos_idx.n_elem; ++p) {
        if((int)pos_idx[p] == master_idx) {
          fluor_order.push_back({unmixed_curr[p], f});
          break;
        }
      }
    }

    // Tie-breaking Sort: Abundance first, then original index for stability
    std::sort(fluor_order.begin(), fluor_order.end(),
              [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                if (std::abs(a.first - b.first) > 1e-9) return a.first > b.first;
                return a.second < b.second;
              });

    // Optimization per Fluorophore
    for (auto const& pair_data : fluor_order) {
      size_t f = pair_data.second;
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

          // If the residual error is lower with this variant, save it
          if (t_err < error_final) {
            error_final = t_err;
            unmixed_curr = t_unmix;
            resid = t_resid;
            resid_norm = std::sqrt(arma::dot(resid, resid));
            cell_spectra_final.row(master_idx) = spectra_curr.row(row_in_curr);
          } else {
            spectra_curr.row(row_in_curr) = original_row;
          }
        }
      }
    }
    // Final unmix with all fluors and optimized spectra
    final_output.row(i) = arma::solve(cell_spectra_final.t(), cell_raw.t(), arma::solve_opts::fast).t();
  }

  return final_output;
}
