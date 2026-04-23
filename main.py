import data_parser
import frontend
import backend
import evaluation
import numpy as np

def main():
    print("=== START PLANAR MONOCULAR SLAM ===\n")
    
    # 1. Reading
    print("[1/4] Data reading...")
    dataset = data_parser.load_all_data("./data")
    
    # 2. Initialization
    print("\n[2/4] Bootstrapping (Initial Traingulation)...")
    initial_map = frontend.run_bootstrap(dataset)
    
    # 3. Optimization
    print("\n[3/4] Bundle Adjustment...")
    optimized_poses, optimized_map = backend.run_bundle_adjustment(dataset, initial_map)
    
    # GT dictionary for evaluation
    pose_ids = dataset['trajectory']['pose_ids']
    gt_data = dataset['trajectory']['ground_truth']
    gt_poses_dict = {pose_ids[i]: gt_data[i] for i in range(len(pose_ids))}

    common_keys = sorted(set(optimized_poses.keys()).intersection(gt_poses_dict.keys()))
    est_xy = np.array([optimized_poses[k][:2] for k in common_keys])
    gt_xy = np.array([gt_poses_dict[k][:2] for k in common_keys])

    R_align, t_align = evaluation.align_se2_points(est_xy, gt_xy)
    aligned_poses = evaluation.apply_se2_alignment_to_poses(optimized_poses, R_align, t_align)
    rmse_abs_traj = evaluation.compute_absolute_trajectory_rmse(aligned_poses, gt_poses_dict)

    odom_data = dataset['trajectory']['odometry']
    odom_poses_dict = {pose_ids[i]: odom_data[i] for i in range(len(pose_ids))}
    
    # 4. Evaluation
    print("\n[4/4] Error Evaluation (Compared to Ground Truth)...")
    rmse_t, rmse_r = evaluation.evaluate_poses(optimized_poses, gt_poses_dict)
    rmse_map = evaluation.evaluate_map(optimized_map, dataset['world_map'])

    reproj_before = evaluation.compute_mean_reprojection_error(
    odom_poses_dict,
    initial_map,
    dataset['measurements'],
    dataset['camera_params']['K'],
    dataset['camera_params']['transform']
    )

    reproj_mean_after, reproj_median_after, n_after = evaluation.compute_reprojection_stats(
    optimized_poses,
    optimized_map,
    dataset['measurements'],
    dataset['camera_params']['K'],
    dataset['camera_params']['transform']
    )
    
    print(f"   -> Initial Landmark: {len(initial_map)}")
    print(f"   -> Final Landmark:   {len(optimized_map)}")
    print(f"   -> RMSE Translation:  {rmse_t:.4f} metri")
    print(f"   -> RMSE Rotation:    {rmse_r:.4f} radianti")
    print(f"   -> RMSE Map:        {rmse_map:.4f} metri")
    print(f"   -> Reproj mean after BA:    {reproj_mean_after:.3f} px")
    print(f"   -> Reproj median after BA:  {reproj_median_after:.3f} px")
    print(f"   -> Observations used after BA: {n_after}")
    print(f"   -> RMSE Absolute Trajectory: {rmse_abs_traj:.4f} meters")
    print("\nGeneration Graph")
    evaluation.plot_results(
    optimized_poses,
    gt_poses_dict,
    optimized_map,
    dataset['world_map'],
    odom_poses_dict=odom_poses_dict
)

if __name__ == "__main__":
    main()