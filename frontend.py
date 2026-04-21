import numpy as np
import cv2
import geometry_utils
import data_parser

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
    
    # 1. We collect the "history" of each landmark
    point_history = {}
    
    for meas in measurements:
        odom_pose = meas['odom_pose'] 
        cam_in_world = geometry_utils.get_camera_pose_in_world(odom_pose, cam_transform)
        
        for i, pt_id in enumerate(meas['point_ids']):
            pixel = meas['image_points'][i]
            
            if pt_id not in point_history:
                point_history[pt_id] = []
            point_history[pt_id].append((cam_in_world, pixel))
            
    #2. Let's triangulate
    initial_map = {}
    
    for pt_id, observations in point_history.items():
        if len(observations) >= 2:
            cam1, pixel1 = observations[0]
            cam2, pixel2 = observations[-1]
            
            # Let's calculate the 3x4 projection matrices
            P1 = geometry_utils.get_projection_matrix(cam1, K)
            P2 = geometry_utils.get_projection_matrix(cam2, K)
            
            pt1 = np.array(pixel1, dtype=np.float32).reshape(2, 1)
            pt2 = np.array(pixel2, dtype=np.float32).reshape(2, 1)
            
            # Triangulation (returns homogeneous 4x1 coordinates)
            X_homog = cv2.triangulatePoints(P1, P2, pt1, pt2)
            
            # Convert from homogeneous to 3D (dividing by the last row W)
            X_3d = X_homog[:3] / X_homog[3]
            # Let's make this a 1D array (x, y, z)
            X_3d = X_3d.flatten() 
            
            # 3. Validity filter
            # Let's transform the 3D point into the reference system of the first camera
            # to check that it is in front of the lens
            world_to_cam1 = geometry_utils.inverse_transform(cam1)
            X_homog_cam1 = world_to_cam1 @ np.append(X_3d, 1.0)
            z_in_cam = X_homog_cam1[2]
            
            if z_near < z_in_cam < z_far:
                initial_map[pt_id] = X_3d
                
    return initial_map

