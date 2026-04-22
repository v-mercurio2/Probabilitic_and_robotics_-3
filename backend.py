import numpy as np
from scipy.optimize import least_squares
import geometry_utils
import data_parser
from scipy.sparse import lil_matrix
import frontend

def pack_state(delta_poses_dict, points_dict):
    """
    Squash delta-poses and landmarks into a single 1D vector.
    """
    pose_keys = sorted(list(delta_poses_dict.keys()))
    point_keys = sorted(list(points_dict.keys()))

    pose_array = np.array([delta_poses_dict[k] for k in pose_keys]).flatten()
    point_array = np.array([points_dict[k] for k in point_keys]).flatten()

    state_vector = np.concatenate((pose_array, point_array))
    return state_vector, pose_keys, point_keys

def unpack_state(state_vector, pose_keys, point_keys):
    """
    Reconstructs delta-pose and landmark from 1D vector.
    """
    n_poses = len(pose_keys)
    poses_flat = state_vector[:n_poses * 3]
    points_flat = state_vector[n_poses * 3:]

    delta_poses_dict = {
        pose_keys[i]: poses_flat[i * 3 : i * 3 + 3]
        for i in range(n_poses)
    }
    points_dict = {
        point_keys[i]: points_flat[i * 3 : i * 3 + 3]
        for i in range(len(point_keys))
    }

    return delta_poses_dict, points_dict

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def relative_pose_2d(pose_a, pose_b):
    """
    Returns the relative motion from pose_a to pose_b expressed in the local frame of pose_a.
    Output: [dx_local, dy_local, dtheta]
    """
    xa, ya, tha = pose_a
    xb, yb, thb = pose_b

    dx_world = xb - xa
    dy_world = yb - ya

    c = np.cos(tha)
    s = np.sin(tha)

    dx_local =  c * dx_world + s * dy_world
    dy_local = -s * dx_world + c * dy_world
    dtheta = wrap_angle(thb - tha)

    return np.array([dx_local, dy_local, dtheta], dtype=float)

def compose_pose_from_odom(odom_pose, delta_pose):
    """
    Reconstructs the absolute pose as:
    pose = odom_pose + delta_pose
    """
    x = odom_pose[0] + delta_pose[0]
    y = odom_pose[1] + delta_pose[1]
    theta = wrap_angle(odom_pose[2] + delta_pose[2])
    return np.array([x, y, theta], dtype=float)

def compute_residuals(state_vector, pose_keys, point_keys, measurements, K, cam_transform,
                      odom_poses_dict, odom_constraints,
                      w_prior=100.0,
                      w_odom=np.array([0.01, 0.01, 0.01]),
                      w_reproj=2.5):
    """
    Total residuals:
    1) prior on the delta of the first pose
    2) relative odometry constraints between consecutive poses
    3) landmark reprojection errors
    """
    delta_poses_dict, points_dict = unpack_state(state_vector, pose_keys, point_keys)

    # I reconstruct the absolute poses: pose = odom + delta
    poses_dict = {}
    for k in pose_keys:
        poses_dict[k] = compose_pose_from_odom(odom_poses_dict[k], delta_poses_dict[k])

    residuals = []

    # 1. Prior on the delta of the first pose
    delta_0 = delta_poses_dict[pose_keys[0]].copy()
    delta_0[2] = wrap_angle(delta_0[2])
    residuals.extend((w_prior * delta_0).tolist())

    # 2. Relative odometric constraints
    for k0, k1, rel_odom in odom_constraints:
        if k0 not in poses_dict or k1 not in poses_dict:
            continue

        rel_est = relative_pose_2d(poses_dict[k0], poses_dict[k1])
        odom_err = rel_est - rel_odom
        odom_err[2] = wrap_angle(odom_err[2])

        residuals.extend((w_odom * odom_err).tolist())
    

    # 3. Reprojection errors
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

            pt_3d = points_dict[pt_id]
            pixel_meas = meas['image_points'][i]

            pt_3d_homog = np.append(pt_3d, 1.0)
            pt_proj_homog = P @ pt_3d_homog

            # Fixed length of the residue vector
            if pt_proj_homog[2] <= 1e-6:
                residuals.append(10.0)
                residuals.append(10.0)
                continue

            u_proj = pt_proj_homog[0] / pt_proj_homog[2]
            v_proj = pt_proj_homog[1] / pt_proj_homog[2]

            residuals.append(w_reproj * (u_proj - pixel_meas[0]))
            residuals.append(w_reproj * (v_proj - pixel_meas[1]))

    return np.array(residuals, dtype=float)

def build_sparsity_matrix(pose_keys, point_keys, measurements, odom_constraints, residuals_length):
    n_poses = len(pose_keys)
    n_points = len(point_keys)

    n_pose_params = n_poses * 3
    n_point_params = n_points * 3
    n_params = n_pose_params + n_point_params

    pose_idx_map = {k: i for i, k in enumerate(pose_keys)}
    point_idx_map = {k: i for i, k in enumerate(point_keys)}

    A = lil_matrix((residuals_length, n_params), dtype=int)

    res_idx = 0

    # 1. Prior on the first pose (3 residues)
    A[res_idx:res_idx+3, 0:3] = 1
    res_idx += 3

    # 2. Odometric constraints (3 residues per constraint)
    for k0, k1, _ in odom_constraints:
        if k0 not in pose_idx_map or k1 not in pose_idx_map:
            continue

        idx0 = pose_idx_map[k0] * 3
        idx1 = pose_idx_map[k1] * 3

        A[res_idx:res_idx+3, idx0:idx0+3] = 1
        A[res_idx:res_idx+3, idx1:idx1+3] = 1
        res_idx += 3

    # 3. Reprojection residues (2 residues per visible point)
    for meas in measurements:
        seq = meas['seq']
        if seq not in pose_idx_map:
            continue

        pose_idx = pose_idx_map[seq]
        pose_col_start = pose_idx * 3
        pose_col_end = pose_col_start + 3

        for pt_id in meas['point_ids']:
            if pt_id not in point_idx_map:
                continue

            point_idx = point_idx_map[pt_id]
            point_col_start = n_pose_params + point_idx * 3
            point_col_end = point_col_start + 3

            # Entrambe le variabili (posa e landmark) influenzano l'errore del pixel
            A[res_idx:res_idx+2, pose_col_start:pose_col_end] = 1
            A[res_idx:res_idx+2, point_col_start:point_col_end] = 1
            res_idx += 2

    return A

def filter_bad_landmarks(optimized_poses, optimized_map, measurements, K, cam_transform,
                         max_mean_reproj_error=10.0, min_valid_obs=2, max_distance=20.0):
    """
    It only keeps landmarks that:
    - are not too far from the origin,
    - have at least min_valid_obs valid observations with positive depth,
    - have average reprojection error below the threshold.
    """
    filtered_map = {}

    for pt_id, pt_3d in optimized_map.items():
        if np.linalg.norm(pt_3d) > max_distance:
            continue

        errors = []
        valid_obs = 0

        for meas in measurements:
            seq = meas['seq']
            if seq not in optimized_poses:
                continue

            ids = meas['point_ids']
            matches = np.where(ids == pt_id)[0]
            if len(matches) == 0:
                continue

            pose_se2 = optimized_poses[seq]
            cam_in_world = geometry_utils.get_camera_pose_in_world(pose_se2, cam_transform)
            P = geometry_utils.get_projection_matrix(cam_in_world, K)

            Xh = np.append(pt_3d, 1.0)
            proj = P @ Xh
            if proj[2] <= 1e-6:
                continue

            uv = proj[:2] / proj[2]

            for idx in matches:
                pixel_meas = meas['image_points'][idx]
                err = np.linalg.norm(uv - pixel_meas)
                errors.append(err)
                valid_obs += 1

        if valid_obs >= min_valid_obs and len(errors) > 0:
            mean_err = np.mean(errors)
            if mean_err <= max_mean_reproj_error:
                filtered_map[pt_id] = pt_3d

    return filtered_map

def run_bundle_adjustment(dataset, initial_map):
    """
    Performs Total Least Squares optimization.
    """
    measurements = dataset['measurements']
    camera_params = dataset['camera_params']

    pose_ids = dataset['trajectory']['pose_ids']
    odom_data = dataset['trajectory']['odometry']

    # Dictionary of absolute odometric poses
    odom_poses_dict = {}
    for i in range(len(pose_ids)):
        odom_poses_dict[pose_ids[i]] = odom_data[i]

    # Initial delta = zero
    initial_deltas = {}
    for k in pose_ids:
        initial_deltas[k] = np.zeros(3, dtype=float)

    # Relative odometric constraints
    odom_constraints = []
    for i in range(len(pose_ids) - 1):
        k0 = pose_ids[i]
        k1 = pose_ids[i + 1]
        rel_odom = relative_pose_2d(odom_data[i], odom_data[i + 1])
        odom_constraints.append((k0, k1, rel_odom))

    print(f"   -> Optimization {len(initial_deltas)} poses and {len(initial_map)} landmarks...")

    state_vector, pose_keys, point_keys = pack_state(initial_deltas, initial_map)
    
    #We need to calculate how many "errors" (residuals) we will produce to dimension the matrix
    print("   -> Calculate error size and sparsity matrix...")
    initial_residuals = compute_residuals(
    state_vector, pose_keys, point_keys, measurements,
    camera_params['K'], camera_params['transform'],
    odom_poses_dict, odom_constraints
    )

    jac_sparsity = build_sparsity_matrix(
        pose_keys, point_keys, measurements, odom_constraints, len(initial_residuals)
    )
        
    print("   -> Start optimizartion...")
    res = least_squares(
        compute_residuals, 
        state_vector, 
        jac_sparsity=jac_sparsity, 
        args=(
        pose_keys, point_keys, measurements,
        camera_params['K'], camera_params['transform'],
        odom_poses_dict, odom_constraints
        ),
        method='trf',
        loss='linear',
        f_scale=5.0,
        verbose=2,
        max_nfev=300
    )
    
    print("\n   -> End optimization!")

    optimized_deltas, optimized_map = unpack_state(res.x, pose_keys, point_keys)

    # I reconstruct final absolute poses
    optimized_poses = {}
    for k in pose_keys:
        optimized_poses[k] = compose_pose_from_odom(odom_poses_dict[k], optimized_deltas[k])

    filtered_map = filter_bad_landmarks(
        optimized_poses,
        optimized_map,
        measurements,
        camera_params['K'],
        camera_params['transform'],
        max_mean_reproj_error=10.0, 
        min_valid_obs=2,
        max_distance=15.0
    )

    print(f"   -> Landmarks after quality filter: {len(filtered_map)} / {len(optimized_map)}")

    return optimized_poses, filtered_map

