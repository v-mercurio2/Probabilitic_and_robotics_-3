import data_parser
import frontend
import backend
import evaluation

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
    
    # 4. Evaluation
    print("\n[4/4] Error evaluation (Compared to Ground Truth)...")
    rmse_t, rmse_r = evaluation.evaluate_poses(optimized_poses, gt_poses_dict)
    rmse_map = evaluation.evaluate_map(optimized_map, dataset['world_map'])
    
    print(f"   -> RMSE Traslation: {rmse_t:.4f} metri")
    print(f"   -> RMSE Rotation:   {rmse_r:.4f} radianti")
    print(f"   -> RMSE Map:       {rmse_map:.4f} metri")
    
    print("\nGeneration Graph ")
    evaluation.plot_results(optimized_poses, gt_poses_dict, optimized_map, dataset['world_map'])

if __name__ == "__main__":
    main()