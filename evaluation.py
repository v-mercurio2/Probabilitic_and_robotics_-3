import numpy as np
import matplotlib.pyplot as plt
import geometry_utils

def evaluate_poses(optimized_poses, gt_poses_dict):
    """
    Calculate the SE(2) error between the estimated poses and the Ground Truth.
    """
    rot_errors = []
    trans_errors = []
    
    pose_keys = sorted(list(optimized_poses.keys()))
    
    for i in range(len(pose_keys) - 1):
        k0 = pose_keys[i]
        k1 = pose_keys[i+1]
        
        # We skip if the GT is missing
        if k0 not in gt_poses_dict or k1 not in gt_poses_dict:
            continue
            
        # Estimated poses
        T_0 = geometry_utils.pose_se2_to_se3(optimized_poses[k0])
        T_1 = geometry_utils.pose_se2_to_se3(optimized_poses[k1])
        
        # Pose Ground Truth
        GT_0 = geometry_utils.pose_se2_to_se3(gt_poses_dict[k0])
        GT_1 = geometry_utils.pose_se2_to_se3(gt_poses_dict[k1])
        
        # Let's calculate the relative motions
        rel_T = geometry_utils.inverse_transform(T_0) @ T_1
        rel_GT = geometry_utils.inverse_transform(GT_0) @ GT_1
        
        # Error SE(2)
        error_T = geometry_utils.inverse_transform(rel_T) @ rel_GT
        
        # Extract errors using indices (in Python we start at 0)
        # error_T(2,1) and error_T(1,1) become error_T[1,0] and error_T[0,0]
        rot_err = np.arctan2(error_T[1, 0], error_T[0, 0])
        rot_errors.append(rot_err)
        
        # Translation error (RMSE on X and Y)
        t_err = error_T[0:2, 3] 
        trans_errors.append(np.linalg.norm(t_err))
        
    rmse_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rmse_rot = np.sqrt(np.mean(np.square(rot_errors)))
    
    return rmse_trans, rmse_rot

def evaluate_map(optimized_map, world_map):
    """
    Calculate the total RMSE between the estimated and actual 3D points.
    """
    errors = []
    for pt_id, est_pt in optimized_map.items():
        if pt_id in world_map:
            gt_pt = world_map[pt_id]
            # Euclidean distance between the estimated point and the true one
            dist = np.linalg.norm(est_pt - gt_pt)
            errors.append(dist)
            
    if not errors:
        return 0.0
        
    rmse_map = np.sqrt(np.mean(np.square(errors)))
    return rmse_map

def plot_results(optimized_poses, gt_poses_dict, optimized_map, world_map):
    """
    Draw the 2D map (top view) with the trajectory and landmarks.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Draw the points on the map (Ground Truth in grey, Estimation in blue)
    gt_pts = np.array(list(world_map.values()))
    est_pts = np.array(list(optimized_map.values()))
    
    if len(gt_pts) > 0:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lightgray', label='Mappa Reale (GT)', s=10)
    if len(est_pts) > 0:
        plt.scatter(est_pts[:, 0], est_pts[:, 1], c='blue', label='Mappa Stimata', s=10, alpha=0.6)
        
    # 2. Disegniamo le traiettorie
    gt_traj = np.array([gt_poses_dict[k] for k in sorted(gt_poses_dict.keys())])
    est_traj = np.array([optimized_poses[k] for k in sorted(optimized_poses.keys())])
    
    if len(gt_traj) > 0:
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'k--', label='Traiettoria GT', linewidth=2)
    if len(est_traj) > 0:
        plt.plot(est_traj[:, 0], est_traj[:, 1], 'r-', label='Traiettoria Stimata (SLAM)', linewidth=2)
        
    plt.title("Planar Monocular SLAM - Risults")
    plt.xlabel("X (metri)")
    plt.ylabel("Y (metri)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Mantiene le proporzioni corrette tra X e Y
    plt.show()