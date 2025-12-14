/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // ====================================================================================
  // IDEA 2: Additional outlier rejection based on IMU consistency and spatial depth
  // ====================================================================================
  apply_outlier_rejection_idea2(state, feature_vec);

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
// Appending to UpdaterMSCKF.cpp - add this content at the end of the file

void UpdaterMSCKF::apply_outlier_rejection_idea2(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &features) {

  // Early exit if no features to process
  if (features.empty()) {
    return;
  }

  // Configurable thresholds
  const double MIN_DEPTH = 0.1;                // meters (close range for indoor)
  const double MAX_DEPTH = 100.0;              // meters
  const double MAX_REPROJECTION_ERROR = 20.0;   // pixels (Stage 1) - relaxed for indoor
  const double DEPTH_MEDIAN_RATIO = 3.0;       // spatial consistency threshold (Stage 2)
  const size_t MIN_FEATURES_FOR_MEDIAN = 5;    // minimum features for statistical check

  // Get current IMU state
  Eigen::Vector3d p_IinG = state->_imu->pos();
  Eigen::Matrix<double, 3, 3> R_GtoI = quat_2_Rot(state->_imu->quat());

  // Use set to avoid duplicate marking
  std::set<size_t> indexes_to_remove;
  std::vector<std::pair<size_t, double>> valid_feature_depths;

  // Stage 1 counters
  int stage1_rejected = 0;

  // ---------------------------------------------------------
  // Stage 1: IMU Consistency Check (Reprojection Error)
  // Reject features with high reprojection error
  // ---------------------------------------------------------
  for (size_t i = 0; i < features.size(); i++) {
    auto feat = features[i];

    // Skip if feature doesn't have triangulated position
    // Note: p_FinG.norm() < 0.01 means either uninitialized (zero) or too close to origin
    // Triangulation should set p_FinG to non-zero if successful
    if (feat->p_FinG.norm() < 0.01) {
      if (i < 5) {
        PRINT_DEBUG("[OUTLIER REJ]: Feature %zu skipped - no triangulated position (norm=%.6f)\n", 
                    feat->featid, feat->p_FinG.norm());
      }
      continue;
    }

    Eigen::Vector3d p_FinG = feat->p_FinG;
    double max_reproj_error = 0.0;
    bool has_valid_observation = false;

    // Check reprojection error for all camera observations
    for (const auto &camid_times : feat->timestamps) {
      size_t cam_id = camid_times.first;
      const std::vector<double> &timestamps = camid_times.second;

      if (timestamps.empty()) continue;

      // Get camera extrinsics (IMU to Camera transform)
      auto calib_it = state->_calib_IMUtoCAM.find(cam_id);
      if (calib_it == state->_calib_IMUtoCAM.end()) {
        continue;
      }
      auto calib_cam = calib_it->second;
      Eigen::Matrix<double, 3, 3> R_ItoC = calib_cam->Rot();
      Eigen::Vector3d p_IinC = calib_cam->pos();

      // Get camera intrinsics
      auto intrinsics_it = state->_cam_intrinsics_cameras.find(cam_id);
      if (intrinsics_it == state->_cam_intrinsics_cameras.end()) {
        continue;
      }
      auto cam_intrinsics = intrinsics_it->second;

      // Loop through each timestamp for this camera
      for (size_t m = 0; m < timestamps.size(); m++) {
        double timestamp = timestamps[m];

        // Get the IMU clone pose at this timestamp
        auto clone_it = state->_clones_IMU.find(timestamp);
        if (clone_it == state->_clones_IMU.end()) {
          continue;
        }
        auto clone_imu = clone_it->second;
        Eigen::Vector3d p_IinG = clone_imu->pos();
        Eigen::Matrix<double, 3, 3> R_GtoI = quat_2_Rot(clone_imu->quat());

        // Transform feature from Global -> IMU -> Camera frame
        Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
        Eigen::Vector3d p_FinC = R_ItoC * p_FinI + p_IinC;

        // Skip if behind camera
        if (p_FinC(2) <= 0) continue;

        // Normalize the 3D point
        Eigen::Vector2d uv_norm;
        uv_norm << p_FinC(0) / p_FinC(2), p_FinC(1) / p_FinC(2);

        // Distort the normalized coordinates to get pixel coordinates
        Eigen::Vector2d uv_dist;
        uv_dist = cam_intrinsics->distort_d(uv_norm);

        // Get the observation at this index
        if (m >= feat->uvs.at(cam_id).size()) continue;
        Eigen::VectorXf uv_obs_vec = feat->uvs.at(cam_id).at(m);
        if (uv_obs_vec.rows() < 2) continue;
        Eigen::Vector2d uv_obs = uv_obs_vec.head<2>().cast<double>();
        
        // Calculate reprojection error
        double reproj_error = (uv_dist - uv_obs).norm();
        max_reproj_error = std::max(max_reproj_error, reproj_error);
        has_valid_observation = true;
      }
    }

    // Reject if reprojection error is too high
    if (has_valid_observation && max_reproj_error > MAX_REPROJECTION_ERROR) {
      indexes_to_remove.insert(i);
      stage1_rejected++;
      if (stage1_rejected <= 3) { // Print first 3 rejected features
        PRINT_DEBUG("[OUTLIER REJ] Stage 1: Feat %zu rejected (reproj_error=%.2f > %.1f)\n",
                    feat->featid, max_reproj_error, MAX_REPROJECTION_ERROR);
      }
    }
  }

  PRINT_INFO("[OUTLIER REJ]: Stage 1 rejected %d features (reprojection error > %.1f pixels)\n",
             stage1_rejected, MAX_REPROJECTION_ERROR);

  // ---------------------------------------------------------
  // Collect depths for all features using simple depth calculation
  // ---------------------------------------------------------
  for (size_t i = 0; i < features.size(); i++) {
    auto feat = features[i];

    // Skip if already marked for deletion in Stage 1
    if (indexes_to_remove.find(i) != indexes_to_remove.end()) {
      continue;
    }

    // Skip if feature doesn't have triangulated position
    if (feat->p_FinG.norm() < 0.01) {
      continue;
    }

    // Use the feature's existing triangulated position
    Eigen::Vector3d p_FinG = feat->p_FinG;

    // Calculate average depth across all camera observations
    double total_depth = 0.0;
    int valid_cam_count = 0;

    for (const auto &camid_times : feat->timestamps) {
      size_t cam_id = camid_times.first;
      const std::vector<double> &timestamps = camid_times.second;

      // Get camera extrinsics (IMU to Camera transform)
      auto calib_it = state->_calib_IMUtoCAM.find(cam_id);
      if (calib_it == state->_calib_IMUtoCAM.end()) {
        continue;
      }
      auto calib_cam = calib_it->second;
      Eigen::Matrix<double, 3, 3> R_ItoC = calib_cam->Rot();
      Eigen::Vector3d p_IinC = calib_cam->pos();

      // Use the most recent timestamp for depth calculation
      if (timestamps.empty()) continue;
      double timestamp = timestamps.back();

      // Get the IMU clone pose at this timestamp
      auto clone_it = state->_clones_IMU.find(timestamp);
      if (clone_it == state->_clones_IMU.end()) {
        continue;
      }
      auto clone_imu = clone_it->second;
      Eigen::Vector3d p_IinG = clone_imu->pos();
      Eigen::Matrix<double, 3, 3> R_GtoI = quat_2_Rot(clone_imu->quat());

      // Transform feature from Global -> IMU -> Camera frame
      Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
      Eigen::Vector3d p_FinC = R_ItoC * p_FinI + p_IinC;

      // Check depth validity
      double depth = p_FinC(2);
      
      // Debug: print depth values to understand rejection
      if (i < 5) { // Only print first 5 features to avoid spam
        PRINT_DEBUG("[DEPTH CHECK] Feat %zu, Cam %zu: depth=%.3f (p_FinC=[%.3f, %.3f, %.3f])\n",
                    feat->featid, cam_id, depth, p_FinC(0), p_FinC(1), p_FinC(2));
      }
      
      if (depth < MIN_DEPTH || depth > MAX_DEPTH) {
        if (i < 5) {
          PRINT_DEBUG("  -> REJECTED: depth %.3f outside [%.1f, %.1f]\n", depth, MIN_DEPTH, MAX_DEPTH);
        }
        continue;
      }

      // Accumulate depth for averaging
      total_depth += depth;
      valid_cam_count++;
    }

    PRINT_INFO("Valid Cam Count %d\n", valid_cam_count) 

    // Store valid feature with average depth
    if (valid_cam_count > 0) {
      double avg_depth = total_depth / valid_cam_count;
      valid_feature_depths.push_back(std::make_pair(i, avg_depth));
      if (i < 5) {
        PRINT_DEBUG("[DEPTH CHECK] Feat %zu: avg_depth=%.3f (from %d cameras)\n", 
                    feat->featid, avg_depth, valid_cam_count);
      }
    } else if (i < 5) {
      PRINT_DEBUG("[DEPTH CHECK] Feat %zu: NO VALID DEPTHS (valid_cam_count=0)\n", feat->featid);
    }
  }
  
  PRINT_INFO("[OUTLIER REJ]: Found %d features with valid depths (after Stage 1)\n", 
             (int)valid_feature_depths.size());

  // ---------------------------------------------------------
  // Stage 2: Spatial Depth Consistency (Median Filter)
  // ---------------------------------------------------------
  int stage2_rejected = 0;
    
  if (valid_feature_depths.size() > MIN_FEATURES_FOR_MEDIAN) {
    // Extract depths and calculate median
    std::vector<double> depths_only;
    depths_only.reserve(valid_feature_depths.size());
    for (const auto &pair : valid_feature_depths) {
      depths_only.push_back(pair.second);
    }

    std::sort(depths_only.begin(), depths_only.end());
    double median_depth = depths_only[depths_only.size() / 2];

    PRINT_INFO("[OUTLIER REJ]: Median depth = %.2f meters (from %zu features)\n", 
               median_depth, valid_feature_depths.size());

    // Check each valid feature against median
    // Remove features with depth significantly different from median
    for (const auto &pair : valid_feature_depths) {
      size_t idx = pair.first;
      double depth = pair.second;

      // Rejection condition: depth outside [median/3, median*3]
      if (depth > median_depth * DEPTH_MEDIAN_RATIO || depth < median_depth / DEPTH_MEDIAN_RATIO) {
        indexes_to_remove.insert(idx);
        stage2_rejected++;
      }
    }

    PRINT_INFO("[OUTLIER REJ]: Stage 2 rejected %d features (spatial consistency)\n", stage2_rejected);
  }

  // ---------------------------------------------------------
  // Mark features for deletion and remove from vector
  // ---------------------------------------------------------
  int removed_count = 0;
  int total_before = features.size();

  // Mark features as to_delete
  for (size_t idx : indexes_to_remove) {
    if (idx < features.size()) {
      features[idx]->to_delete = true;
      removed_count++;
    }
  }

  // Remove marked features from vector
  auto it = features.begin();
  while (it != features.end()) {
    if ((*it)->to_delete) {
      it = features.erase(it);
    } else {
      it++;
    }
  }

  PRINT_INFO("[OUTLIER REJ]: Total removed %d out of %d features (Stage 1: %d, Stage 2: %d)\n",
              removed_count, total_before, stage1_rejected, stage2_rejected);
}
