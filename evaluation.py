import numpy as np
import matplotlib.pyplot as plt
import geometry_utils

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

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
        t_err = error_T[0:2, 3] # Prende le prime due righe dell'ultima colonna
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
            # Distanza Euclidea tra il punto stimato e quello vero
            dist = np.linalg.norm(est_pt - gt_pt)
            errors.append(dist)
            
    if not errors:
        return 0.0
        
    rmse_map = np.sqrt(np.mean(np.square(errors)))
    return rmse_map

def evaluate_map_aligned(aligned_map, world_map):
    errors = []

    for pt_id, est_pt in aligned_map.items():
        if pt_id in world_map:
            gt_pt = world_map[pt_id]
            dist = np.linalg.norm(est_pt - gt_pt)
            errors.append(dist)

    if not errors:
        return 0.0

    errors = np.array(errors)
    return float(np.sqrt(np.mean(errors**2)))

def compute_mean_reprojection_error(poses_dict, points_dict, measurements, K, cam_transform):
    errors = []

    for meas in measurements:
        seq = meas['seq']
        if seq not in poses_dict:
            continue

        pose_se2 = poses_dict[seq]
        cam_in_world = geometry_utils.get_camera_pose_in_world(pose_se2, cam_transform)
        P = geometry_utils.get_projection_matrix(cam_in_world, K)

        for i, pt_id in enumerate(meas['point_ids']):
            if pt_id not in points_dict:
                continue

            Xh = np.append(points_dict[pt_id], 1.0)
            proj = P @ Xh
            if proj[2] <= 1e-8:
                continue

            uv = proj[:2] / proj[2]
            err = np.linalg.norm(uv - meas['image_points'][i])
            errors.append(err)

    if len(errors) == 0:
        return 0.0

    return float(np.mean(errors))

def align_se2_points(est_xy, gt_xy):
    """
    Find the 2D rotation-translation that aligns est_xy to gt_xy in a least-squares sense.
    Returns R (2x2) and t (2,).
    """
    est_centroid = np.mean(est_xy, axis=0)
    gt_centroid = np.mean(gt_xy, axis=0)

    est_centered = est_xy - est_centroid
    gt_centered = gt_xy - gt_centroid

    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Corregge eventuale riflessione
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = gt_centroid - R @ est_centroid
    return R, t

def apply_se2_alignment_to_points(points_dict, R, t):
    """
    Apply the same SE(2) rotation-translation to 3D landmarks.
    Rotate/translate only X,Y; Z remains unchanged.
    """
    aligned = {}

    for k, pt in points_dict.items():
        xy_new = R @ pt[:2] + t
        aligned[k] = np.array([xy_new[0], xy_new[1], pt[2]])

    return aligned


def apply_se2_alignment_to_poses(poses_dict, R, t):
    """
    Apply 2D roto-translation to the poses [x,y,theta].
    """
    aligned = {}

    rot_angle = np.arctan2(R[1, 0], R[0, 0])

    for k, pose in poses_dict.items():
        x, y, theta = pose
        xy_new = R @ np.array([x, y]) + t
        theta_new = wrap_angle(theta + rot_angle)
        aligned[k] = np.array([xy_new[0], xy_new[1], theta_new])

    return aligned


def compute_absolute_trajectory_rmse(aligned_poses, gt_poses_dict):
    """
    Absolute RMSE on the trajectory, after alignment.
    """
    errs = []

    common_keys = sorted(set(aligned_poses.keys()).intersection(gt_poses_dict.keys()))
    for k in common_keys:
        est_xy = aligned_poses[k][:2]
        gt_xy = gt_poses_dict[k][:2]
        errs.append(np.linalg.norm(est_xy - gt_xy))

    if len(errs) == 0:
        return 0.0

    errs = np.array(errs)
    return float(np.sqrt(np.mean(errs**2)))

def plot_results(optimized_poses, gt_poses_dict, optimized_map, world_map, odom_poses_dict=None):
    """
    Draws the 2D map (top view) with the trajectory and landmarks.
    The estimated trajectory is first rigidly aligned in SE(2) to the GT
    for visualization only.
    """
    plt.figure(figsize=(10, 8))

    # 1. GT Points
    gt_pts = np.array(list(world_map.values()))

    # 2. SE(2) alignment of the estimated trajectory to the GT
    common_keys = sorted(set(optimized_poses.keys()).intersection(gt_poses_dict.keys()))
    est_xy = np.array([optimized_poses[k][:2] for k in common_keys])
    gt_xy = np.array([gt_poses_dict[k][:2] for k in common_keys])

    R_align, t_align = align_se2_points(est_xy, gt_xy)
    aligned_poses = apply_se2_alignment_to_poses(optimized_poses, R_align, t_align)
    aligned_map = apply_se2_alignment_to_points(optimized_map, R_align, t_align)

    # 3. Estimated landmarks aligned
    est_pts = np.array(list(aligned_map.values())) if len(aligned_map) > 0 else np.empty((0, 3))

    if len(gt_pts) > 0:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], c='lightgray', label='Real Map (GT)', s=10)
    if len(est_pts) > 0:
        plt.scatter(est_pts[:, 0], est_pts[:, 1], c='blue', label='Estimated Map', s=10, alpha=0.6)

    # 4. Let's draw the trajectories
    gt_traj = np.array([gt_poses_dict[k] for k in sorted(gt_poses_dict.keys())])
    est_traj = np.array([aligned_poses[k] for k in sorted(aligned_poses.keys())])

    if odom_poses_dict is not None:
        odom_xy = np.array([odom_poses_dict[k][:2] for k in common_keys])
        R_odom, t_odom = align_se2_points(odom_xy, gt_xy)
        aligned_odom = apply_se2_alignment_to_poses(odom_poses_dict, R_odom, t_odom)
        odom_traj = np.array([aligned_odom[k] for k in sorted(aligned_odom.keys())])
        if len(odom_traj) > 0:
            plt.plot(odom_traj[:, 0], odom_traj[:, 1], 'g:', label='Odometry', linewidth=2)

    if len(gt_traj) > 0:
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'k--', label='GT Trajectory', linewidth=2)
    if len(est_traj) > 0:
        plt.plot(est_traj[:, 0], est_traj[:, 1], 'r-', label='Estimated Trajectory (SLAM)', linewidth=2)

    plt.title("Planar Monocular SLAM - Results")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def compute_reprojection_stats(poses_dict, points_dict, measurements, K, cam_transform):
    errors = []

    for meas in measurements:
        seq = meas['seq']
        if seq not in poses_dict:
            continue

        pose_se2 = poses_dict[seq]
        cam_in_world = geometry_utils.get_camera_pose_in_world(pose_se2, cam_transform)
        P = geometry_utils.get_projection_matrix(cam_in_world, K)

        for i, pt_id in enumerate(meas['point_ids']):
            if pt_id not in points_dict:
                continue

            Xh = np.append(points_dict[pt_id], 1.0)
            proj = P @ Xh
            if proj[2] <= 1e-8:
                continue

            uv = proj[:2] / proj[2]
            err = np.linalg.norm(uv - meas['image_points'][i])
            errors.append(err)

    if len(errors) == 0:
        return 0.0, 0.0, 0

    errors = np.array(errors)
    return float(np.mean(errors)), float(np.median(errors)), int(len(errors))