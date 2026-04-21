import numpy as np
import glob
import os

def load_world(filepath):
    """
    Read world.dat for final evaluation.
    Expected format: LANDMARK_ID (1) POSITION (2:4)
    """
    # We use np.loadtxt which automatically handles spaces and carriage returns
    data = np.loadtxt(filepath)
    
    world_map = {}
    for row in data:
        landmark_id = int(row[0])
        position = row[1:4] 
        world_map[landmark_id] = position
        
    return world_map

def load_trajectory(filepath):
    """
    Read trajectory.dat, which contains the odometry and ground truth.
    Expected format: POSE_ID (1) ODOMETRY_POSE (2:4) GROUNDTRUTH_POSE (5:7)
    """
    data = np.loadtxt(filepath)
    
    trajectory = {
        'pose_ids': data[:, 0].astype(int),
        'odometry': data[:, 1:4],      # Pose [x, y, theta] stimate/rumorose
        'ground_truth': data[:, 4:7]   # Pose [x, y, theta] reali
    }
    return trajectory

def load_camera_params(filepath):
    """
    Reads the camera.dat file, extracting the intrinsic matrix, the transform,
    the viewing limits (z_near, z_far), and the resolution.
    """
    camera_params = {}
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith("camera matrix:"):
            # The next 3 rows make up the 3x3 matrix K
            matrix_data = []
            for j in range(1, 4):
                row = [float(x) for x in lines[i+j].split()]
                matrix_data.append(row)
            camera_params['K'] = np.array(matrix_data)
            i += 4  
            
        elif line.startswith("cam_transform:"):
            # The next 4 rows make up the 4x4 transformation matrix
            transform_data = []
            for j in range(1, 5):
                row = [float(x) for x in lines[i+j].split()]
                transform_data.append(row)
            camera_params['transform'] = np.array(transform_data)
            i += 5
            
        elif line.startswith("z_near:"):
            camera_params['z_near'] = float(line.split(':')[1].strip())
            i += 1
            
        elif line.startswith("z_far:"):
            camera_params['z_far'] = float(line.split(':')[1].strip())
            i += 1
            
        elif line.startswith("width:"):
            camera_params['width'] = int(line.split(':')[1].strip())
            i += 1
            
        elif line.startswith("height:"):
            camera_params['height'] = int(line.split(':')[1].strip())
            i += 1
            
        else:
            i += 1 
            
    return camera_params

def load_measurements(folder_path):
    """
    Reads all meas-XXXX.dat files knowing their exact structure.
    """
    meas_files = sorted(glob.glob(os.path.join(folder_path, "meas-*.dat")))
    measurements = []
    
    for file in meas_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            
        points_data = []
        header_info = {}
        
        # Let's analyze the file line by line
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue 
                
            # We identify what type of line it is by looking at the first word
            if parts[0] == 'seq:':
                header_info['seq'] = int(parts[1])
                
            elif parts[0] == 'gt_pose:':
                header_info['gt_pose'] = [float(x) for x in parts[1:]]
                
            elif parts[0] == 'odom_pose:':
                header_info['odom_pose'] = [float(x) for x in parts[1:]]
                
            elif parts[0] == 'point':
                # Format: point (0) CURRENT_ID (1) ACTUAL_ID (2) COL (3) ROW (4)
                points_data.append([
                    float(parts[1]), 
                    float(parts[2]), 
                    float(parts[3]), 
                    float(parts[4])
                ])
                
        # If we found some points, we create the dictionary for this measurement
        if points_data:
            points_data = np.array(points_data)
            meas_dict = {
                'filename': os.path.basename(file),
                'seq': header_info.get('seq'),
                'gt_pose': np.array(header_info.get('gt_pose')),
                'odom_pose': np.array(header_info.get('odom_pose')),
                'point_ids': points_data[:, 1].astype(int), 
                'image_points': points_data[:, 2:4]         
            }
            measurements.append(meas_dict)
            
    return measurements

def load_all_data(data_dir):
    """
    Wrapper function that calls all the others and returns a single large dictionary.
    """
    return {
        'world_map': load_world(os.path.join(data_dir, 'world.dat')),
        'trajectory': load_trajectory(os.path.join(data_dir, 'trajectoy.dat')),
        'camera_params': load_camera_params(os.path.join(data_dir, 'camera.dat')),
        'measurements': load_measurements(data_dir)
    }

