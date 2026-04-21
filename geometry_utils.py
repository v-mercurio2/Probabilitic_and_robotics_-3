import numpy as np

def pose_se2_to_se3(pose_se2):
    """
    Converts a 2D robot pose (x, y, theta) into a 4x4 3D homogeneous transformation matrix (SE3).
    The robot lies on the XY plane, so z=0.
    """
    x, y, theta = pose_se2
    
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0, x],
        [np.sin(theta),  np.cos(theta), 0.0, y],
        [0.0,            0.0,           1.0, 0.0],
        [0.0,            0.0,           0.0, 1.0]
    ])

def get_camera_pose_in_world(pose_se2, cam_transform):
    """
    Calculate the 4x4 camera pose in the real world.
    Multiply the robot pose by the fixed camera pose on the robot.
    """
    robot_in_world = pose_se2_to_se3(pose_se2)
    camera_in_world = robot_in_world @ cam_transform
    return camera_in_world

def inverse_transform(T):
    """
    Calculate the inverse of a 4x4 homogeneous transformation matrix.
    Exploit the property of rotation matrices (R^-1 = R^T) for faster and more accurate results.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def get_projection_matrix(camera_in_world, K):
    """
    Calculate the projection matrix P (3x4) for a specific camera pose.
    Formula: P = K * [R | t]
    Where [R | t] is the transformation from WORLD to CAMERA (inverse of the camera pose).
    """
    #1. We find the transformation from World to Camera
    world_to_camera = inverse_transform(camera_in_world)
    
    # 2. We extract only the part [R | t] which is 3x4 (ignore the last row [0, 0, 0, 1])
    Rt = world_to_camera[:3, :]
    
    # 3. Multiply by the intrinsic matrix K
    P = K @ Rt
    return P