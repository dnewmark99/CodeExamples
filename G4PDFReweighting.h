#ifndef G4PDFReweighting_H
#define G4PDFReweighting_H

#include <icetray/IcetrayFwd.h>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/math/special_functions.hpp>

#include <tuple>
#include <vector>
#include <random>
#include <cmath>
#include <map>
#include <algorithm>
#include <mutex>

#include "dataio/I3File.h"
#include "icetray/ctpl.h"
#include "icetray/I3Units.h"
#include "dataclasses/I3Position.h"
#include "simclasses/CCMMCPE.h"
#include "simclasses/PhotonSummary.h"
#include "icetray/CCMPMTKey.h"
#include "analytic-light-yields/LightParameters.h"
#include "analytic-light-yields/G4YieldsPerPMT.h"

template <typename T>
struct G4PDFReweightingJob {
    std::atomic<bool> running = false;
    size_t event_idx = 0;
    std::pair<size_t, size_t> event_ranges;
    std::map<CCMPMTKey, std::vector<T>> * Q11_binned_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q11_binned_square_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q12_binned_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q12_binned_square_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q21_binned_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q21_binned_square_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q22_binned_yields = nullptr;
    std::map<CCMPMTKey, std::vector<T>> * Q22_binned_square_yields = nullptr;
};

template <typename T>
struct G4PDFReweightingResult {
    size_t event_idx = 0;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q11_binned_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q11_binned_square_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q12_binned_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q12_binned_square_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q21_binned_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q21_binned_square_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q22_binned_yields = nullptr;
    std::shared_ptr<std::map<CCMPMTKey, std::vector<T>>> Q22_binned_square_yields = nullptr;
    bool done = false;
};

class G4PDFReweighting {
    ctpl::thread_pool pool;
    size_t max_cached_vertices = (size_t) 3000;
    std::exception_ptr teptr = nullptr;

public:
    G4PDFReweighting();
    template<typename T> void GetAllYields(size_t n_threads,
                                           std::vector<CCMPMTKey> const & keys_to_fit,
                                           std::vector<double> time_bin_edges,
                                           SimulationCoords<T> coords,
                                           G4PhotonSimulationData<T> * data,
                                           G4PhotonTemporarySimulationData<T> * temp_data,
                                           T uv_absorption1,
                                           T uv_absorption2,
                                           T uv_absorption_scaling,
                                           T pmt_qe,
                                           T endcap_qe,
                                           T side_qe,
                                           T norm,
                                           bool fit_z,
                                           bool fit_rayl,
                                           bool scintillation_flag);

};

template<typename T> void ScalePDFReweightBinYields(std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * all_sodium_events,
                                                    simulated_parameters sim_params,
                                                    std::vector<CCMPMTKey> const & keys_to_fit,
                                                    std::vector<double> & time_bin_edges,
                                                    size_t & event_start,
                                                    size_t & event_end,
                                                    T uv_absorption1,
                                                    T uv_absorption2,
                                                    T uv_absorption_scaling,
                                                    T pmt_qe,
                                                    T endcap_qe,
                                                    T side_qe,
                                                    T normalization,
                                                    T alpha_total,
                                                    double alpha0_total,
                                                    std::vector<T> const & Light,
                                                    std::vector<double> const & Light0,
                                                    std::map<CCMPMTKey, std::vector<T>> const & GeV,
                                                    std::map<CCMPMTKey, std::vector<double>> const & GeV0,
                                                    std::vector<double> const & light_time_bin_edges,
                                                    bool scintillation_flag,
                                                    std::map<CCMPMTKey, std::vector<T>> * this_event_binned_yields,
                                                    std::map<CCMPMTKey, std::vector<T>> * this_event_binned_square_yields) {

    size_t n_bins = time_bin_edges.size() - 1;
    double light_bin_width = light_time_bin_edges[1] - light_time_bin_edges[0];
    double data_bin_width = time_bin_edges[1] - time_bin_edges[0];
    // now let's loop over each event in our vector sodium_events
    for(size_t event_it = event_start; event_it < event_end; ++event_it) {

        std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>> const & parsed_event = all_sodium_events->at(event_it);

        // now loop over all pmts we are fitting
        for(size_t k = 0; k < keys_to_fit.size(); k++){
            CCMPMTKey this_key = keys_to_fit.at(k);

            // let's check if this pmt is in our map to save binned events to, if not we add it!
            if (this_event_binned_yields->find(this_key) == this_event_binned_yields->end()) {
                (*this_event_binned_yields)[this_key] = std::vector<T>(n_bins, 0.0);
            }
            if (this_event_binned_square_yields->find(this_key) == this_event_binned_square_yields->end()) {
                (*this_event_binned_square_yields)[this_key] = std::vector<T>(n_bins, 0.0);
            }

            // grab the list of lightweight CCMMCPEs for this key
            if (parsed_event.find(this_key) == parsed_event.end()) {
                continue; // Skip this PMT if it's not present in the event
            }
            std::vector<ccmmcpe_lightweight> const & this_key_ccmmcpe = parsed_event.at(this_key);

            // loop over each lightweight CCMMCPE
            for (size_t m = 0; m < this_key_ccmmcpe.size(); m++){
                ccmmcpe_lightweight const & this_ccmmcpe = this_key_ccmmcpe.at(m);

                double const & distance_travelled_uv = this_ccmmcpe.g4_distance_uv / I3Units::cm;
                double const & original_wavelength = this_ccmmcpe.original_wavelength / I3Units::nanometer;
                double const & detection_wavelength = this_ccmmcpe.wavelength / I3Units::nanometer;
                double const & distance_travelled_vis = this_ccmmcpe.g4_distance_vis / I3Units::cm;

                // for our uv absorption, we are using this paper https://link.springer.com/article/10.1140/epjc/s10052-012-2190-z#Bib1
                // and fitting for scale and a
                // uv_absorption1 == a, uv_absorption_scaling == scaling
                // T(wavelength) = 1 - e^(-a * (wavelength - b))
                // AbsLength(wavelength) = d / ln (1 / T(wavelength))
                // for the units, wavelength are in nm and abs length is in cm

                T d(5.8);
                T b(113.0);
                T uv_scaling = 0.0;
                // based on original wavelength and distance travelled, calculate ratio of uv absoprtion scaling
                if (original_wavelength > b){
                    T uv_scaling0 = pow(1.0 - exp(sim_params.uv_absorption1 * (b - original_wavelength)), distance_travelled_uv / (sim_params.uv_absorption_scaling * d));
                    T vis_scaling0 = pow(1.0 - exp(sim_params.uv_absorption1 * (b - detection_wavelength)), distance_travelled_vis / (sim_params.uv_absorption_scaling * d));
                    T this_uv_scaling = FDpower<T>()(1.0 - exp(uv_absorption1 * (b - original_wavelength)), distance_travelled_uv / (uv_absorption_scaling * d));
                    T this_vis_scaling = FDpower<T>()(1.0 - exp(uv_absorption1 * (b - detection_wavelength)), distance_travelled_vis / (uv_absorption_scaling * d));

                    // now something special...using flat absorption lengths for 140 to 400nm
                    // pmt_qe == absorption length 140 to 230
                    // endcap_qe == absorption length 230 to 320
                    // and side_qe == absorption length 320 to 400
                    if (original_wavelength >= 140.0 and original_wavelength < 200.0){
                        this_uv_scaling =  exp(- distance_travelled_uv / pmt_qe);
                    } else if (original_wavelength >= 200.0 and original_wavelength < 300.0){
                        this_uv_scaling =  exp(- distance_travelled_uv / endcap_qe);
                    } else if (original_wavelength >= 300.0 and original_wavelength < 400.0){
                        this_uv_scaling =  exp(- distance_travelled_uv / side_qe);
                    }

                    // same game for distance visible
                    if (detection_wavelength >= 140.0 and detection_wavelength < 200.0){
                        this_vis_scaling =  exp(- distance_travelled_vis / pmt_qe);
                    } else if (detection_wavelength >= 200.0 and detection_wavelength < 300.0){
                        this_vis_scaling =  exp(- distance_travelled_vis / endcap_qe);
                    } else if (detection_wavelength >= 300.0 and detection_wavelength < 400.0){
                        this_vis_scaling =  exp(- distance_travelled_vis / side_qe);
                    }

                    if (sim_params.uv_abs_140_200 > 0.0){
                        if (original_wavelength >= 140.0 and original_wavelength < 200.0){
                            uv_scaling0 =  exp(- distance_travelled_uv / sim_params.uv_abs_140_200);
                        } else if (original_wavelength >= 200.0 and original_wavelength < 300.0){
                            uv_scaling0 =  exp(- distance_travelled_uv / sim_params.uv_abs_200_300);
                        } else if (original_wavelength >= 300.0 and original_wavelength < 400.0){
                            uv_scaling0 =  exp(- distance_travelled_uv / sim_params.uv_abs_300_400);
                        }

                        // same game for distance visible
                        if (detection_wavelength >= 140.0 and detection_wavelength < 200.0){
                            vis_scaling0 = exp(- distance_travelled_vis / sim_params.uv_abs_140_200);
                        } else if (detection_wavelength >= 200.0 and detection_wavelength < 300.0){
                            vis_scaling0 = exp(- distance_travelled_vis / sim_params.uv_abs_200_300);
                        } else if (detection_wavelength >= 300.0 and detection_wavelength < 400.0){
                            vis_scaling0 = exp(- distance_travelled_vis / sim_params.uv_abs_300_400);
                        }
                    }

                    uv_scaling = (this_uv_scaling * this_vis_scaling) / (uv_scaling0 * vis_scaling0);
                }

                T tpb_qe_scaling = 1.0;

                // almost done, now let's get our reweighting for light + gev
                // set up for not-in-gev case
                // grab light time
                double light_time = this_ccmmcpe.light_time;
                if (light_time < light_time_bin_edges.front() || light_time >= light_time_bin_edges.back()) {
                    continue;
                }

                int light_bin_idx = static_cast<int>((light_time - light_time_bin_edges.front()) / light_bin_width);

                T numerator = (1.0 - alpha_total) * Light.at(light_bin_idx);
                T denominator = (1.0 - alpha0_total) * Light0.at(light_bin_idx);
                // special case if we are in a gev
                if (this_ccmmcpe.in_gev){
                    // grab afterpulse time
                    double afterpulse_time = this_ccmmcpe.afterpulse_time;
                    if (afterpulse_time < light_time_bin_edges.front() || afterpulse_time >= light_time_bin_edges.back()) {
                        continue;
                    }
                    int afterpulse_bin_idx = static_cast<int>((afterpulse_time - light_time_bin_edges.front()) / light_bin_width);

                    numerator = Light.at(light_bin_idx) * GeV.at(this_key).at(afterpulse_bin_idx);
                    denominator = Light0.at(light_bin_idx) * GeV0.at(this_key).at(afterpulse_bin_idx);
                }

                // now grab data time bin!
                double reco_time = this_ccmmcpe.reco_time;
                if (reco_time < time_bin_edges.front() || reco_time >= time_bin_edges.back()) {
                    continue;
                }
                std::vector<double>::const_iterator data_time_it = std::upper_bound(time_bin_edges.cbegin(), time_bin_edges.cend(), reco_time);
                int reco_bin_idx = std::distance(time_bin_edges.cbegin(), data_time_it) - 1;
                // now save
                // dont forget sampled amplitude
                double sampled_amplitude = this_ccmmcpe.amplitude;
                T total_norm = sampled_amplitude;
                if (scintillation_flag){
                    total_norm *= normalization;
                }
                T this_photon_yield = total_norm * uv_scaling * tpb_qe_scaling * (numerator / denominator);

                this_event_binned_yields->at(this_key).at(reco_bin_idx) += this_photon_yield;
                this_event_binned_square_yields->at(this_key).at(reco_bin_idx) += this_photon_yield * this_photon_yield;
            }
        }
    }
}

template<typename T>
void G4PDFReweightingFrameThread(int id,
                 std::vector<CCMPMTKey> const & keys_to_fit,
                 std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q11_events,
                 std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q12_events,
                 std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q21_events,
                 std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q22_events,
                 simulated_parameters Q11_simulated_params,
                 simulated_parameters Q12_simulated_params,
                 simulated_parameters Q21_simulated_params,
                 simulated_parameters Q22_simulated_params,
                 T uv_absorption1,
                 T uv_absorption2,
                 T uv_absorption_scaling,
                 T pmt_qe,
                 T endcap_qe,
                 T side_qe,
                 T norm,
                 T alpha_total,
                 double alpha0_total,
                 std::vector<T> const & Light,
                 std::vector<double> const & Light0,
                 std::map<CCMPMTKey, std::vector<T>> const & GeV,
                 std::map<CCMPMTKey, std::vector<double>> const & GeV0,
                 std::vector<double> const & light_time_bin_edges,
                 std::vector<double> time_bin_edges,
                 G4PDFReweightingJob<T> * job,
                 bool fit_z,
                 bool fit_rayl,
                 bool scintillation_flag,
                 std::array<std::map<CCMPMTKey, std::vector<T>>*, 8> global_yields_results,
                 std::map<CCMPMTKey, std::mutex> * tube_mutex) {

    size_t event_start = job->event_ranges.first;
    size_t event_end = job->event_ranges.second;

    T normalization = norm / Q11_simulated_params.norm;

    // this function reweights our photons by uv abs and tpb qe, and pdfs
    ScalePDFReweightBinYields<T>(Q11_events, Q11_simulated_params, keys_to_fit, time_bin_edges, event_start, event_end,
                                 uv_absorption1, uv_absorption2, uv_absorption_scaling, pmt_qe, endcap_qe, side_qe, normalization,
                                 alpha_total, alpha0_total, Light, Light0, GeV, GeV0, light_time_bin_edges,
                                 scintillation_flag, job->Q11_binned_yields, job->Q11_binned_square_yields);
    if(fit_z) {
        normalization = norm / Q21_simulated_params.norm;
        ScalePDFReweightBinYields<T>(Q21_events, Q21_simulated_params, keys_to_fit, time_bin_edges, event_start, event_end,
                                     uv_absorption1, uv_absorption2, uv_absorption_scaling, pmt_qe, endcap_qe, side_qe, normalization,
                                     alpha_total, alpha0_total, Light, Light0, GeV, GeV0, light_time_bin_edges,
                                     scintillation_flag, job->Q21_binned_yields, job->Q21_binned_square_yields);
    }
    if(fit_rayl) {
        normalization = norm / Q12_simulated_params.norm;
        ScalePDFReweightBinYields<T>(Q12_events, Q12_simulated_params, keys_to_fit, time_bin_edges, event_start, event_end,
                                     uv_absorption1, uv_absorption2, uv_absorption_scaling, pmt_qe, endcap_qe, side_qe, normalization,
                                     alpha_total, alpha0_total, Light, Light0, GeV, GeV0, light_time_bin_edges,
                                     scintillation_flag, job->Q12_binned_yields, job->Q12_binned_square_yields);
    }
    if (fit_z and fit_rayl) {
        normalization = norm / Q22_simulated_params.norm;
        ScalePDFReweightBinYields<T>(Q22_events, Q22_simulated_params, keys_to_fit, time_bin_edges, event_start, event_end,
                                     uv_absorption1, uv_absorption2, uv_absorption_scaling, pmt_qe, endcap_qe, side_qe, normalization,
                                     alpha_total, alpha0_total, Light, Light0, GeV, GeV0, light_time_bin_edges,
                                     scintillation_flag, job->Q22_binned_yields, job->Q22_binned_square_yields);
    }

    // let's save our results binned yields to appropriate quadrant
    for (auto& [key, values] : *(global_yields_results[0])) {
        // aquire a  lock on this key
        std::lock_guard<std::mutex> lock(tube_mutex->at(key));
        CompileResults(key,
                       fit_z,
                       fit_rayl,
                       job,
                       *global_yields_results[0], *global_yields_results[1],
                       *global_yields_results[2], *global_yields_results[3],
                       *global_yields_results[4], *global_yields_results[5],
                       *global_yields_results[6], *global_yields_results[7]
                );
    }
    job->running.store(false);
}

template<typename T>
void G4PDFReweightingRunFrameThread(ctpl::thread_pool & pool,
                    G4PDFReweightingJob<T> * job,
                    std::vector<CCMPMTKey> const & keys_to_fit,
                    T uv_absorption1,
                    T uv_absorption2,
                    T uv_absorption_scaling,
                    T pmt_qe,
                    T endcap_qe,
                    T side_qe,
                    T norm,
                    T alpha_total,
                    std::vector<double> time_bin_edges,
                    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q11_events,
                    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q12_events,
                    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q21_events,
                    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q22_events,
                    simulated_parameters Q11_simulated_params,
                    simulated_parameters Q12_simulated_params,
                    simulated_parameters Q21_simulated_params,
                    simulated_parameters Q22_simulated_params,
                    double alpha0_total,
                    std::vector<T> const & Light,
                    std::vector<double> const & Light0,
                    std::map<CCMPMTKey, std::vector<T>> const & GeV,
                    std::map<CCMPMTKey, std::vector<double>> const & GeV0,
                    std::vector<double> const & light_time_bin_edges,
                    bool fit_z,
                    bool fit_rayl,
                    bool scintillation_flag,
                    std::array<std::map<CCMPMTKey, std::vector<T>>*, 8> global_yields_results,
                    std::map<CCMPMTKey, std::mutex> * tube_mutex) {

    job->running.store(true);
    pool.push(G4PDFReweightingFrameThread<T>,
            std::cref(keys_to_fit),
            Q11_events,
            Q12_events,
            Q21_events,
            Q22_events,
            Q11_simulated_params,
            Q12_simulated_params,
            Q21_simulated_params,
            Q22_simulated_params,
            uv_absorption1,
            uv_absorption2,
            uv_absorption_scaling,
            pmt_qe,
            endcap_qe,
            side_qe,
            norm,
            alpha_total,
            alpha0_total,
            Light,
            Light0,
            GeV,
            GeV0,
            light_time_bin_edges,
            time_bin_edges,
            job,
            fit_z,
            fit_rayl,
            scintillation_flag,
            global_yields_results,
            tube_mutex
    );
}

template<typename T> void CompileResults(CCMPMTKey key,
                                         bool fit_z,
                                         bool fit_rayl,
                                         G4PDFReweightingJob<T> const * job,
                                         std::map<CCMPMTKey, std::vector<T>> & Q11_binned_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q11_binned_square_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q12_binned_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q12_binned_square_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q21_binned_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q21_binned_square_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q22_binned_yields,
                                         std::map<CCMPMTKey, std::vector<T>> & Q22_binned_square_yields){
    size_t size = job->Q11_binned_yields->at(key).size();

    // Use raw pointers for direct access (avoids bound checks)
    auto* Q11_y_ptr  = Q11_binned_yields[key].data();
    auto* Q11_sq_ptr = Q11_binned_square_yields[key].data();
    auto* Q21_y_ptr  = Q21_binned_yields[key].data();
    auto* Q21_sq_ptr = Q21_binned_square_yields[key].data();
    auto* Q12_y_ptr  = Q12_binned_yields[key].data();
    auto* Q12_sq_ptr = Q12_binned_square_yields[key].data();
    auto* Q22_y_ptr  = Q22_binned_yields[key].data();
    auto* Q22_sq_ptr = Q22_binned_square_yields[key].data();

    auto* Q11_y_val  = job->Q11_binned_yields->at(key).data();
    auto* Q11_sq_val = job->Q11_binned_square_yields->at(key).data();

    T * Q21_y_val;
    T * Q21_sq_val;
    T * Q12_y_val;
    T * Q12_sq_val;
    T * Q22_y_val;
    T * Q22_sq_val;

    if (fit_z){
        Q21_y_val  = job->Q21_binned_yields->at(key).data();
        Q21_sq_val = job->Q21_binned_square_yields->at(key).data();
    }

    if (fit_rayl){
        Q12_y_val  = job->Q12_binned_yields->at(key).data();
        Q12_sq_val = job->Q12_binned_square_yields->at(key).data();
    }

    if (fit_z && fit_rayl){
        Q22_y_val  = job->Q22_binned_yields->at(key).data();
        Q22_sq_val = job->Q22_binned_square_yields->at(key).data();
    }

    // Loop unrolling for performance
    size_t b = 0;
    for (; b + 3 < size; b += 4) {
        Q11_y_ptr[b]   += Q11_y_val[b];   Q11_sq_ptr[b]   += Q11_sq_val[b];
        Q11_y_ptr[b+1] += Q11_y_val[b+1]; Q11_sq_ptr[b+1] += Q11_sq_val[b+1];
        Q11_y_ptr[b+2] += Q11_y_val[b+2]; Q11_sq_ptr[b+2] += Q11_sq_val[b+2];
        Q11_y_ptr[b+3] += Q11_y_val[b+3]; Q11_sq_ptr[b+3] += Q11_sq_val[b+3];

        if (fit_z) {
            Q21_y_ptr[b]   += Q21_y_val[b];   Q21_sq_ptr[b]   += Q21_sq_val[b];
            Q21_y_ptr[b+1] += Q21_y_val[b+1]; Q21_sq_ptr[b+1] += Q21_sq_val[b+1];
            Q21_y_ptr[b+2] += Q21_y_val[b+2]; Q21_sq_ptr[b+2] += Q21_sq_val[b+2];
            Q21_y_ptr[b+3] += Q21_y_val[b+3]; Q21_sq_ptr[b+3] += Q21_sq_val[b+3];
        }
        if (fit_rayl) {
            Q12_y_ptr[b]   += Q12_y_val[b];   Q12_sq_ptr[b]   += Q12_sq_val[b];
            Q12_y_ptr[b+1] += Q12_y_val[b+1]; Q12_sq_ptr[b+1] += Q12_sq_val[b+1];
            Q12_y_ptr[b+2] += Q12_y_val[b+2]; Q12_sq_ptr[b+2] += Q12_sq_val[b+2];
            Q12_y_ptr[b+3] += Q12_y_val[b+3]; Q12_sq_ptr[b+3] += Q12_sq_val[b+3];
        }
        if (fit_z && fit_rayl) {
            Q22_y_ptr[b]   += Q22_y_val[b];   Q22_sq_ptr[b]   += Q22_sq_val[b];
            Q22_y_ptr[b+1] += Q22_y_val[b+1]; Q22_sq_ptr[b+1] += Q22_sq_val[b+1];
            Q22_y_ptr[b+2] += Q22_y_val[b+2]; Q22_sq_ptr[b+2] += Q22_sq_val[b+2];
            Q22_y_ptr[b+3] += Q22_y_val[b+3]; Q22_sq_ptr[b+3] += Q22_sq_val[b+3];
        }
    }

    // Handle remainder
    for (; b < size; ++b) {
        Q11_y_ptr[b] += Q11_y_val[b];
        Q11_sq_ptr[b] += Q11_sq_val[b];

        if (fit_z) {
            Q21_y_ptr[b] += Q21_y_val[b];
            Q21_sq_ptr[b] += Q21_sq_val[b];
        }
        if (fit_rayl) {
            Q12_y_ptr[b] += Q12_y_val[b];
            Q12_sq_ptr[b] += Q12_sq_val[b];
        }
        if (fit_z && fit_rayl) {
            Q22_y_ptr[b] += Q22_y_val[b];
            Q22_sq_ptr[b] += Q22_sq_val[b];
        }
    }
}

template<typename T> void G4PDFReweighting::GetAllYields(size_t n_threads,
                                                         std::vector<CCMPMTKey> const & keys_to_fit,
                                                         std::vector<double> time_bin_edges,
                                                         SimulationCoords<T> coords,
                                                         G4PhotonSimulationData<T> * data,
                                                         G4PhotonTemporarySimulationData<T> * temp_data,
                                                         T uv_absorption1,
                                                         T uv_absorption2,
                                                         T uv_absorption_scaling,
                                                         T pmt_qe,
                                                         T endcap_qe,
                                                         T side_qe,
                                                         T norm,
                                                         bool fit_z,
                                                         bool fit_rayl,
                                                         bool scintillation_flag) {

    // make mutext object for every tube
    std::map<CCMPMTKey, std::mutex> tube_mutex;
    for (auto& key : keys_to_fit) {
        tube_mutex[key];  // Create a mutex for each key
    }

    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q11_events = nullptr;
    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q12_events = nullptr;
    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q21_events = nullptr;
    std::vector<std::map<CCMPMTKey, std::vector<ccmmcpe_lightweight>>> const * Q22_events = nullptr;

    // set up light profile for reweighting
    std::vector<T> Light;
    std::vector<double> Light0;

    // the same for scintillation and cherenkov light
    std::map<CCMPMTKey, std::vector<T>> GeV = temp_data->GeV;
    std::map<CCMPMTKey, std::vector<double>> GeV0 = temp_data->GeV0;
    T alpha_total = temp_data->alpha_total;
    double alpha0_total = temp_data->alpha0_total;
    std::vector<double> light_time_bin_edges = temp_data->light_times_bin_edges_double;
    size_t n_time_bins = time_bin_edges.size() - 1;

    if (scintillation_flag){
        Q11_events = &(data->collated_events_scintillation.at(coords.Q11));
        Q12_events = &(data->collated_events_scintillation.at(coords.Q12));
        Q21_events = &(data->collated_events_scintillation.at(coords.Q21));
        Q22_events = &(data->collated_events_scintillation.at(coords.Q22));
        Light = temp_data->S;
        Light0 = temp_data->S0;
    } else {
        Q11_events = &(data->collated_events_cherenkov.at(coords.Q11));
        Q12_events = &(data->collated_events_cherenkov.at(coords.Q12));
        Q21_events = &(data->collated_events_cherenkov.at(coords.Q21));
        Q22_events = &(data->collated_events_cherenkov.at(coords.Q22));
        Light.resize(light_time_bin_edges.size()-1, T(1.0));
        Light0.resize(light_time_bin_edges.size()-1, 1.0);
    }

    simulated_parameters Q11_simulated_params = data->sim_params.at(coords.Q11);
    simulated_parameters Q12_simulated_params = data->sim_params.at(coords.Q12);
    simulated_parameters Q21_simulated_params = data->sim_params.at(coords.Q21);
    simulated_parameters Q22_simulated_params = data->sim_params.at(coords.Q22);

    // set up maps to save binned yields from each quadrant
    std::map<CCMPMTKey, std::vector<T>> Q11_binned_yields;
    std::map<CCMPMTKey, std::vector<T>> Q11_binned_square_yields;
    std::map<CCMPMTKey, std::vector<T>> Q12_binned_yields;
    std::map<CCMPMTKey, std::vector<T>> Q12_binned_square_yields;
    std::map<CCMPMTKey, std::vector<T>> Q21_binned_yields;
    std::map<CCMPMTKey, std::vector<T>> Q21_binned_square_yields;
    std::map<CCMPMTKey, std::vector<T>> Q22_binned_yields;
    std::map<CCMPMTKey, std::vector<T>> Q22_binned_square_yields;

    std::array<std::map<CCMPMTKey, std::vector<T>>*, 8> global_yields_results = {
        &Q11_binned_yields, &Q11_binned_square_yields,
        &Q12_binned_yields, &Q12_binned_square_yields,
        &Q21_binned_yields, &Q21_binned_square_yields,
        &Q22_binned_yields, &Q22_binned_square_yields,
    };

    // make way to keep track of if our final results are initialized or not per pmt
    for (size_t k = 0; k < keys_to_fit.size(); k++){
        Q11_binned_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q11_binned_square_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q12_binned_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q12_binned_square_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q21_binned_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q21_binned_square_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q22_binned_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
        Q22_binned_square_yields[keys_to_fit.at(k)] = std::vector<T>(n_time_bins, T(0.0));
    }

    // will be used for multi-threading our simulation jobs
    std::deque<std::shared_ptr<G4PDFReweightingJob<T>>> jobs;
    std::deque<G4PDFReweightingJob<T> *> free_jobs;
    std::deque<G4PDFReweightingJob<T> *> running_jobs;
    std::deque<G4PDFReweightingResult<T>> results;
    size_t min_vertex_idx = 0;

    // set up our num threads
    size_t num_threads;
    if (n_threads == 0){
        num_threads = std::thread::hardware_concurrency();
    } else{
        num_threads = n_threads;
    }
    pool.resize(num_threads);

    // now let's chunk up our event to give each thread a range of events

    size_t n_events_per_thread = Q11_events->size() / num_threads;
    size_t left_over_events = Q11_events->size() - (num_threads * n_events_per_thread);

    std::vector<std::pair<size_t, size_t>> thread_event_ranges;

    size_t event_range_start = 0;
    size_t event_range_end = event_range_start;
    size_t accounted_for_left_over_events = 0;
    for (size_t thread_it = 0; thread_it < num_threads; thread_it++){

        event_range_start = event_range_end;
        event_range_end = event_range_start + n_events_per_thread; // each thread gets mandatory n_events_per_thread

        // check if we've doled out all of the left over events yet
        if (accounted_for_left_over_events < left_over_events){
            event_range_end += 1;
            accounted_for_left_over_events += 1;
        }

        // save idxs
        thread_event_ranges.push_back(std::make_pair(event_range_start, event_range_end));
    }

    // now let's loop over our pre-made lists of sodium events for each thread
    for(size_t thread_it = 0; thread_it < thread_event_ranges.size(); ++thread_it) {
        while(true) {
            // Check if any jobs have finished
            for(int i=int(running_jobs.size())-1; i>=0; --i) {
			    if (teptr) {
				    try{
					    std::rethrow_exception(teptr);
				    }
				    catch(const std::exception &ex)
				    {
					    std::cerr << "Thread exited with exception: " << ex.what() << "\n";
				    }
			    }
			    if(not running_jobs.at(i)->running.load()) {
				    G4PDFReweightingJob<T> * job = running_jobs.at(i);
                    running_jobs.erase(running_jobs.begin() + i);
                    free_jobs.push_back(job);
                    results.at(job->event_idx - min_vertex_idx).done = true;
                } else {
                    G4PDFReweightingJob<T> * job = running_jobs.at(i);
                }
            }

            // Check for any done results and push the corresponding frames
            size_t results_done = 0;
            for(size_t i=0; i<results.size(); ++i) {
                if(results.at(i).done) {
                    // let's save our results binned yields to appropriate quadrant
                    // now reset our results object
                    results.at(i).Q11_binned_yields = nullptr;
                    results.at(i).Q11_binned_square_yields = nullptr;
                    results.at(i).Q12_binned_yields = nullptr;
                    results.at(i).Q12_binned_square_yields = nullptr;
                    results.at(i).Q21_binned_yields = nullptr;
                    results.at(i).Q21_binned_square_yields = nullptr;
                    results.at(i).Q22_binned_yields = nullptr;
                    results.at(i).Q22_binned_square_yields = nullptr;
                    results.at(i).event_idx = 0;
                    results_done += 1;
                } else {
                    break;
                }
            }
            if(results_done > 0) {
                results.erase(results.begin(), results.begin() + results_done);
                min_vertex_idx += results_done;
            }

            // Attempt to queue up a new job for the frame
            G4PDFReweightingJob<T> * job = nullptr;

            if(free_jobs.size() > 0) {
                job = free_jobs.front();
                job->running.store(false);
                free_jobs.pop_front();
            } else if(running_jobs.size() < num_threads) {
                std::shared_ptr<G4PDFReweightingJob<T>> shared_job = std::make_shared<G4PDFReweightingJob<T>>();
                jobs.push_back(shared_job);
                job = shared_job.get();
                job->running.store(false);
            }

            if(job != nullptr and results.size() < max_cached_vertices) {
                job->running.store(true);
                running_jobs.push_back(job);
                job->event_idx = thread_it;
                // add event ranges for this job
                job->event_ranges = thread_event_ranges.at(thread_it);
                // add emtpy maps to store binned yields for each quadrant
                results.emplace_back();
                results.back().event_idx = job->event_idx;
                results.back().Q11_binned_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q11_binned_square_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q12_binned_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q12_binned_square_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q21_binned_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q21_binned_square_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q22_binned_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                results.back().Q22_binned_square_yields = std::make_shared<std::map<CCMPMTKey, std::vector<T>>>();
                job->Q11_binned_yields = results.back().Q11_binned_yields.get();
                job->Q11_binned_square_yields = results.back().Q11_binned_square_yields.get();
                job->Q12_binned_yields = results.back().Q12_binned_yields.get();
                job->Q12_binned_square_yields = results.back().Q12_binned_square_yields.get();
                job->Q21_binned_yields = results.back().Q21_binned_yields.get();
                job->Q21_binned_square_yields = results.back().Q21_binned_square_yields.get();
                job->Q22_binned_yields = results.back().Q22_binned_yields.get();
                job->Q22_binned_square_yields = results.back().Q22_binned_square_yields.get();
                results.back().done = false;
                G4PDFReweightingRunFrameThread<T>(pool, job, keys_to_fit,
                                                        uv_absorption1,
                                                        uv_absorption2,
                                                        uv_absorption_scaling,
                                                        pmt_qe,
                                                        endcap_qe,
                                                        side_qe,
                                                        norm,
                                                        alpha_total,
                                                        time_bin_edges,
                                                        Q11_events, Q12_events, Q21_events, Q22_events,
                                                        Q11_simulated_params, Q12_simulated_params, Q21_simulated_params, Q22_simulated_params,
                                                        alpha0_total,
                                                        Light,
                                                        Light0,
                                                        GeV,
                                                        GeV0,
                                                        light_time_bin_edges,
                                                        fit_z, fit_rayl, scintillation_flag,
                                                        global_yields_results,
                                                        &tube_mutex);

                break;
            } else if(job != nullptr) {
                free_jobs.push_back(job);
            }
        }
    }
    // final check for any running jobs
    while(running_jobs.size() > 0) {
        // Check if any jobs have finished
        for(int i=int(running_jobs.size())-1; i>=0; --i) {
            if(not running_jobs.at(i)->running.load()) {
                G4PDFReweightingJob<T> * job = running_jobs.at(i);
                running_jobs.erase(running_jobs.begin() + i);
                free_jobs.push_back(job);
                results.at(job->event_idx - min_vertex_idx).done = true;
            }
        }
        // Check for any done results and push the corresponding frames
        size_t results_done = 0;
        for(size_t i=0; i<results.size(); ++i) {
            if(results.at(i).done) {
                // now reset our results object
                results.at(i).Q11_binned_yields = nullptr;
                results.at(i).Q11_binned_square_yields = nullptr;
                results.at(i).Q12_binned_yields = nullptr;
                results.at(i).Q12_binned_square_yields = nullptr;
                results.at(i).Q21_binned_yields = nullptr;
                results.at(i).Q21_binned_square_yields = nullptr;
                results.at(i).Q22_binned_yields = nullptr;
                results.at(i).Q22_binned_square_yields = nullptr;
                results.at(i).event_idx = 0;
                results_done += 1;
            } else {
                break;
            }
        }
        if(results_done > 0) {
            results.erase(results.begin(), results.begin() + results_done);
            min_vertex_idx += results_done;
        }
    }

    // ok we finished binning!!! time to interpolate!!!!
    // iterate through all the keys in our binned yields
    for (auto it = Q11_binned_yields.begin(); it != Q11_binned_yields.end(); ++it) {
        std::vector<T> const & this_key_Q11 = it->second;
        std::vector<T> const & this_key_Q11_squared = Q11_binned_square_yields.at(it->first);

        if(fit_z and not fit_rayl) {
            std::vector<T> const & this_key_Q21 = Q21_binned_yields.at(it->first);
            std::vector<T> const & this_key_Q21_squared = Q21_binned_square_yields.at(it->first);

            if (scintillation_flag){
                temp_data->binned_yields_scintillation[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_scintillation[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q11, this_key_Q21, temp_data->binned_yields_scintillation[it->first]);
                GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q11_squared, this_key_Q21_squared, temp_data->binned_square_yields_scintillation[it->first]);
            } else {
                temp_data->binned_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q11, this_key_Q21, temp_data->binned_yields_cherenkov[it->first]);
                GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q11_squared, this_key_Q21_squared, temp_data->binned_square_yields_cherenkov[it->first]);
            }
        } else if(fit_rayl and not fit_z) {
            std::vector<T> const & this_key_Q12 = Q12_binned_yields.at(it->first);
            std::vector<T> const & this_key_Q12_squared = Q12_binned_square_yields.at(it->first);

            if (scintillation_flag){
                temp_data->binned_yields_scintillation[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_scintillation[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, this_key_Q11, this_key_Q12, temp_data->binned_yields_scintillation[it->first]);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, this_key_Q11_squared, this_key_Q12_squared, temp_data->binned_square_yields_scintillation[it->first]);
            } else {
                temp_data->binned_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, this_key_Q11, this_key_Q12, temp_data->binned_yields_cherenkov[it->first]);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, this_key_Q11_squared, this_key_Q12_squared, temp_data->binned_square_yields_cherenkov[it->first]);
            }
        } else if (fit_z and fit_rayl) {
            std::vector<T> const & this_key_Q21 = Q21_binned_yields.at(it->first);
            std::vector<T> const & this_key_Q21_squared = Q21_binned_square_yields.at(it->first);
            std::vector<T> const & this_key_Q12 = Q12_binned_yields.at(it->first);
            std::vector<T> const & this_key_Q12_squared = Q12_binned_square_yields.at(it->first);
            std::vector<T> const & this_key_Q22 = Q22_binned_yields.at(it->first);
            std::vector<T> const & this_key_Q22_squared = Q22_binned_square_yields.at(it->first);

            std::vector<T> interpolate_x_y_below (this_key_Q11.size(), 0.0);
            std::vector<T> interpolate_x_y_below_squared (this_key_Q11.size(), 0.0);
            GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q11, this_key_Q21, interpolate_x_y_below);
            GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q11_squared, this_key_Q21_squared, interpolate_x_y_below_squared);

            // now interpolate in the x direction
            std::vector<T> interpolate_x_y_above (this_key_Q11.size(), 0.0);
            std::vector<T> interpolate_x_y_above_squared (this_key_Q11.size(), 0.0);
            GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q12, this_key_Q22, interpolate_x_y_above);
            GeneralInterpolation<T, T>()(coords.z, coords.z_below, coords.z_above, this_key_Q12_squared, this_key_Q22_squared, interpolate_x_y_above_squared);

            // now interpolate in the y direction
            if (scintillation_flag){
                temp_data->binned_yields_scintillation[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_scintillation[it->first] = std::vector<T>(this_key_Q11_squared.size(), 0.0);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, interpolate_x_y_below, interpolate_x_y_above, temp_data->binned_yields_scintillation[it->first]);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, interpolate_x_y_below_squared, interpolate_x_y_above_squared, temp_data->binned_square_yields_scintillation[it->first]);
            } else {
                temp_data->binned_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11_squared.size(), 0.0);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, interpolate_x_y_below, interpolate_x_y_above, temp_data->binned_yields_cherenkov[it->first]);
                GeneralInterpolation<T, T>()(coords.rayl, coords.rayl_below, coords.rayl_above, interpolate_x_y_below_squared, interpolate_x_y_above_squared, temp_data->binned_square_yields_cherenkov[it->first]);
            }
        } else {
            if (scintillation_flag){
                temp_data->binned_yields_scintillation[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_scintillation[it->first] = std::vector<T>(this_key_Q11_squared.size(), 0.0);
            } else {
                temp_data->binned_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11.size(), 0.0);
                temp_data->binned_square_yields_cherenkov[it->first] = std::vector<T>(this_key_Q11_squared.size(), 0.0);
            }
            for(size_t i = 0; i < this_key_Q11.size(); ++i) {
                if (scintillation_flag){
                    temp_data->binned_yields_scintillation[it->first].at(i) = this_key_Q11.at(i);
                    temp_data->binned_square_yields_scintillation[it->first].at(i) = this_key_Q11_squared.at(i);
                } else {
                    temp_data->binned_yields_cherenkov[it->first].at(i) = this_key_Q11.at(i);
                    temp_data->binned_square_yields_cherenkov[it->first].at(i) = this_key_Q11_squared.at(i);
                }
            }
        }
    }
}

#endif
