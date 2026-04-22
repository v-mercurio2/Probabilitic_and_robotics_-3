import numpy as np
import cv2
import geometry_utils
import data_parser

def camera_center(T):
    return T[:3, 3]

def select_best_observation_pair(observations, min_baseline=0.60):
    """
    Choose the pair of observations with the maximum baseline.
    """
    best_pair = None
    best_dist = -1.0

    n = len(observations)
    for i in range(n):
        for j in range(i + 1, n):
            cam_i, pix_i = observations[i]
            cam_j, pix_j = observations[j]

            ci = camera_center(cam_i)
            cj = camera_center(cam_j)
            dist = np.linalg.norm(ci - cj)

            if dist > best_dist:
                best_dist = dist
                best_pair = (cam_i, pix_i, cam_j, pix_j)

    if best_pair is None or best_dist < min_baseline:
        return None

    return best_pair

def project_point(P, X_3d):
    Xh = np.append(X_3d, 1.0)
    proj = P @ Xh
    if proj[2] <= 1e-8:
        return None
    return proj[:2] / proj[2]

def run_bootstrap(dataset):
    """
    Performs bootstrapping: triangulates the initial 3D points using noisy odometry.
    """
    camera_params = dataset['camera_params']
    K = camera_params['K']
    cam_transform = camera_params['transform']
    z_near = camera_params.get('z_near', 0.0)
    z_far = camera_params.get('z_far', 10.0) # Un default se non c'è
    
    measurements = dataset['measurements']
    
    #1. We collect the "story" of each landmark
    point_history = {}
    
    for meas in measurements:
        odom_pose = meas['odom_pose'] # Usiamo quella rumorosa, come chiede il testo!
        cam_in_world = geometry_utils.get_camera_pose_in_world(odom_pose, cam_transform)
        
        for i, pt_id in enumerate(meas['point_ids']):
            pixel = meas['image_points'][i]
            
            if pt_id not in point_history:
                point_history[pt_id] = []
            point_history[pt_id].append((cam_in_world, pixel))
            
    #2. Let's triangulate
    initial_map = {}
    
    for pt_id, observations in point_history.items():
        if len(observations) < 2:
            continue

        best = select_best_observation_pair(observations, min_baseline=0.60)
        if best is None:
            continue

        cam1, pixel1, cam2, pixel2 = best

        P1 = geometry_utils.get_projection_matrix(cam1, K)
        P2 = geometry_utils.get_projection_matrix(cam2, K)

        pt1 = np.array(pixel1, dtype=np.float32).reshape(2, 1)
        pt2 = np.array(pixel2, dtype=np.float32).reshape(2, 1)

        X_homog = cv2.triangulatePoints(P1, P2, pt1, pt2)

        # Avoid numerically dangerous divisions
        if abs(X_homog[3, 0]) < 1e-10:
            continue

        X_3d = (X_homog[:3] / X_homog[3]).flatten()

        # Coarse anti-outlier filter: landmarks too far away
        if np.linalg.norm(X_3d) > 12.0:
            continue

        # Depth control in both chambers
        Xh = np.append(X_3d, 1.0)
        z1 = (geometry_utils.inverse_transform(cam1) @ Xh)[2]
        z2 = (geometry_utils.inverse_transform(cam2) @ Xh)[2]

        if not (z_near < z1 < z_far and z_near < z2 < z_far):
            continue

        # Initial reprojection error check
        uv1 = project_point(P1, X_3d)
        uv2 = project_point(P2, X_3d)

        if uv1 is None or uv2 is None:
            continue

        err1 = np.linalg.norm(uv1 - pixel1)
        err2 = np.linalg.norm(uv2 - pixel2)

        if err1 > 3.0 or err2 > 3.0:
            continue

        initial_map[pt_id] = X_3d

    if len(initial_map) > 0:
        pts = np.array(list(initial_map.values()))
        print(f"Bootstrap landmarks valid: {len(initial_map)}")
        print(f"X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
        print(f"Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
        print(f"Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
        print(f"Max distance from the origin: {np.linalg.norm(pts, axis=1).max():.3f}")

    return initial_map

