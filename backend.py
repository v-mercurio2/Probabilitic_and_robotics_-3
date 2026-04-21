import numpy as np
from scipy.optimize import least_squares
import geometry_utils
import data_parser
from scipy.sparse import lil_matrix
import frontend

def pack_state(poses_dict, points_dict):
    """
    It takes the pose and map dictionaries and "compresses" them into a single 1D vector for Scipy.
    It also returns the keys so the dictionaries can be reconstructed later.
    """
    pose_keys = sorted(list(poses_dict.keys()))
    point_keys = sorted(list(points_dict.keys()))
    
    pose_array = np.array([poses_dict[k] for k in pose_keys]).flatten()
    point_array = np.array([points_dict[k] for k in point_keys]).flatten()
    
    state_vector = np.concatenate((pose_array, point_array))
    return state_vector, pose_keys, point_keys

def unpack_state(state_vector, pose_keys, point_keys):
    """
    It takes Scipy's 1D vector and reconstructs the original dictionaries.
    """
    n_poses = len(pose_keys)
    poses_flat = state_vector[:n_poses * 3]
    points_flat = state_vector[n_poses * 3:]
    
    poses_dict = {pose_keys[i]: poses_flat[i*3 : i*3+3] for i in range(n_poses)}
    points_dict = {point_keys[i]: points_flat[i*3 : i*3+3] for i in range(len(point_keys))}
    
    return poses_dict, points_dict

def compute_residuals(state_vector, pose_keys, point_keys, measurements, K, cam_transform, initial_pose_0):
    """
    THE ERROR FUNCTION: Calculates the distance between the real pixels and the projected ones.
    """
    poses_dict, points_dict = unpack_state(state_vector, pose_keys, point_keys)
    residuals = []
    
    # 1. ANCHORING CONSTRAINT (Prior)
    # We pin the very first pose (pose_0) by adding a very high error if it tries to move.
    # This prevents global drift of the map (Gauge Freedom).
    pose_0_current = poses_dict[pose_keys[0]]
    residuals.extend((pose_0_current - initial_pose_0) * 10000.0) 

    # 2. CALCULATION OF THE REPROJECTION ERROR
    for meas in measurements:
        seq = meas['seq']
        if seq not in poses_dict:
            continue
            
        pose_se2 = poses_dict[seq]
        cam_in_world = geometry_utils.get_camera_pose_in_world(pose_se2, cam_transform)
        P = geometry_utils.get_projection_matrix(cam_in_world, K)
        
        for i, pt_id in enumerate(meas['point_ids']):
            if pt_id in points_dict:
                pt_3d = points_dict[pt_id]
                pixel_meas = meas['image_points'][i]
                
                # Let's project the 3D point onto the virtual camera
                pt_3d_homog = np.append(pt_3d, 1.0)
                pt_proj_homog = P @ pt_3d_homog
                
                # If the point ends up behind the camera, we give a huge penalty
                if pt_proj_homog[2] <= 0:
                    residuals.extend([1000.0, 1000.0])
                    continue
                    
                u_proj = pt_proj_homog[0] / pt_proj_homog[2]
                v_proj = pt_proj_homog[1] / pt_proj_homog[2]
                
                # Difference between where the pixel SHOULD be and where it ACTUALLY IS
                residuals.append(u_proj - pixel_meas[0])
                residuals.append(v_proj - pixel_meas[1])
                
    return np.array(residuals)

def build_sparsity_matrix(pose_keys, point_keys, measurements, residuals_length):
    """
    Creates a matrix that tells the optimizer which variables are related to which errors.
    - Rows: The total number of errors (the residues that the compute_residuals function outputs)
    - Columns: The total number of variables (poses + points)
    """
    n_poses = len(pose_keys)
    n_points = len(point_keys)
    
    n_pose_params = n_poses * 3
    n_point_params = n_points * 3
    n_params = n_pose_params + n_point_params
    
    # Let's create a quick map to find the indexes
    pose_idx_map = {k: i for i, k in enumerate(pose_keys)}
    point_idx_map = {k: i for i, k in enumerate(point_keys)}
    
    # lil_matrix is ​​perfect for filling up quickly
    A = lil_matrix((residuals_length, n_params), dtype=int)
    
    #1. Anchoring the first pose (The first 3 residues depend on the first 3 variables)
    A[0:3, 0:3] = 1
    
    res_idx = 3
    for meas in measurements:
        seq = meas['seq']
        if seq not in pose_idx_map:
            continue
            
        pose_idx = pose_idx_map[seq]
        pose_col_start = pose_idx * 3
        pose_col_end = pose_col_start + 3
        
        for pt_id in meas['point_ids']:
            if pt_id in point_idx_map:
                point_idx = point_idx_map[pt_id]
                point_col_start = n_pose_params + point_idx * 3
                point_col_end = point_col_start + 3
                
                # These 2 residues (pixel u, pixel v) depend on the current Pose and the current Point
                A[res_idx:res_idx+2, pose_col_start:pose_col_end] = 1
                A[res_idx:res_idx+2, point_col_start:point_col_end] = 1
                
                res_idx += 2
                
    return A

def run_bundle_adjustment(dataset, initial_map):
    """
    Performs Total Least Squares optimization.
    """
    measurements = dataset['measurements']
    camera_params = dataset['camera_params']
    
    # We prepare the dictionary of initial poses (Noisy Odometry)
    # We reconstruct it from the raw data
    initial_poses = {}
    pose_ids = dataset['trajectory']['pose_ids']
    odom_data = dataset['trajectory']['odometry']
    for i in range(len(pose_ids)):
        initial_poses[pose_ids[i]] = odom_data[i]
        
    print(f"   -> Ottimizzo {len(initial_poses)} pose e {len(initial_map)} landmarks...")
    
    # Let's compress everything into the mega-vector
    state_vector, pose_keys, point_keys = pack_state(initial_poses, initial_map)
    initial_pose_0 = initial_poses[pose_keys[0]].copy() 
    
    # We need to calculate how many "errors" (residuals) we will produce to dimension the matrix
    print("   -> Calculate error size and sparsity matrix (TURBO ON)...")
    initial_residuals = compute_residuals(
        state_vector, pose_keys, point_keys, measurements, 
        camera_params['K'], camera_params['transform'], initial_pose_0
    )
    
    jac_sparsity = build_sparsity_matrix(pose_keys, point_keys, measurements, len(initial_residuals))
    
    print("   -> Starting optimization...")
    res = least_squares(
        compute_residuals, 
        state_vector, 
        jac_sparsity=jac_sparsity, # ECCO IL SEGRETO DELLA VELOCITÀ!
        args=(pose_keys, point_keys, measurements, camera_params['K'], camera_params['transform'], initial_pose_0),
        method='trf',
        loss='huber',
        verbose=2,
        max_nfev=200
    )
    
    print("\n   -> Optimization completed!")
    
    # Scompattiamo il risultato nei formati comodi
    optimized_poses, optimized_map = unpack_state(res.x, pose_keys, point_keys)
    
    return optimized_poses, optimized_map
