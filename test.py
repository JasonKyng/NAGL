import os
import torch
import yaml
import argparse
from tqdm import trange
from argparse import ArgumentParser, Action 

import csv

from utils.utils import get_dataset_info
from utils.detection import run_anomaly_detection
from utils.post_eval_process import eval_finished_run

from models.model import NAGL

class IntListAction(Action):
    """
    Define a custom action to always return a list. 
    This allows --shots 1 to be treated as a list of one element [1]. 
    """
    def __call__(self, namespace, values):
        if not isinstance(values, list):
            values = [values]
        setattr(namespace, self.dest, values)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--image_size", type=int, default=512, help="image size")

    parser.add_argument("--save_path", type=str, default='output/checkpoint', help='path to save results')
    
    parser.add_argument("--backbone_name", type=str, default='dinov2_vits14', help="the name of encoder")
    parser.add_argument("--num_learnable_proxies", type=int, default=25, help="number of learnable queries")
    parser.add_argument("--n_shots", nargs='+', type=int, default=[1], #action=IntListAction,
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--a_shots", nargs='+', type=int, default=[1], #action=IntListAction,
                        help="List of shots to evaluate. Full-shot scenario is -1.")

    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction, help="Evaluate anomaly detection performance.")
    parser.add_argument("--eval_segm", default=False, action=argparse.BooleanOptionalAction, help="Evaluate anomaly segmentation performance.")

    parser.add_argument("--tag", type=str, default='base', help="Optional tag for the saving directory.")

    parser.add_argument('--local_rank', type=int, default=0, help='number of cpu threads to use during batch generation')
    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()
    
    print(f"Requested to run {len(args.n_shots)} (different) shot(s):", args.n_shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    objects, object_anomalies = get_dataset_info(args.dataset)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    
    args.model_name = args.backbone_name

    if args.just_seed != None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)
    
    result_root = f"results/{args.model_name}_{args.image_size}"
    delete_anomlay_maps = True

    for n_shot in list(args.n_shots):
        for a_shot in list(args.a_shots):
            args.n_shot = n_shot
            args.a_shot = a_shot
            model = NAGL(args)
            model.to(args.device)
            # Load checkpoint
            if a_shot>0:
                ckpt = torch.load(f'{args.save_path}/n_{n_shot}_a_{a_shot}_best.pth')
                print(f"Loading checkpoint from {args.save_path}/n_{n_shot}_a_{a_shot}_best.pth")
                for name, param in ckpt.items():
                    if 'module' in name:
                        name = name[7:]
                    # set the model parameters by name
                    if name in model.state_dict().keys():
                        model.state_dict()[name].copy_(param)
                        print(f"Loaded {name} from checkpoint.")
                    else:
                        print(f"Warning: {name} not in model state_dict.")
            model.eval()

            results_dir_suffix = f"{n_shot}-n_shot_{a_shot}-a_shot"
            results_dir = os.path.join(result_root, args.tag, args.dataset, results_dir_suffix)
            # else:
                # results_dir = os.path.join(result_root, args.dataset, results_dir_suffix)
            plots_dir = results_dir
            os.makedirs(f"{results_dir}", exist_ok=True)

            # save arguments to file
            with open(f"{results_dir}/args.yaml", "w") as f:
                yaml.dump(vars(args), f)

            print("Results will be saved to", results_dir)
        
            for seed in seeds:
                print(f"=========== N-Shot = {n_shot}, A-Shot = {a_shot}, Seed = {seed} ===========")
                
                if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                    print(f"Results for N-Shot = {n_shot}, A-Shot = {a_shot}, Seed = {seed} already exist. Skipping.")
                    continue
                else:
                    timeit_file = results_dir + "/time_measurements.csv"
                    with open(timeit_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])

                        for object_name in objects:             
                            anomaly_scores, time_inference = run_anomaly_detection(
                                                                                    args,
                                                                                    model,
                                                                                    object_name,
                                                                                    data_root = args.data_root, 
                                                                                    object_anomalies = object_anomalies,
                                                                                    plots_dir = plots_dir,
                                                                                    seed = seed,
                                                                                    save_patch_dists = args.eval_clf, # save patch distances for detection evaluation
                                                                                    save_tiffs = args.eval_segm)      # save anomaly maps as tiffs for segmentation evaluation
                            
                            # write anomaly scores and inference times to file
                            for counter, sample in enumerate(anomaly_scores.keys()):
                                anomaly_score = anomaly_scores[sample]
                                inference_time = time_inference[sample]
                                writer.writerow([object_name, sample, f"{anomaly_score:.5f}", f"{inference_time:.5f}"])

                    # read inference times from file
                    with open(timeit_file, 'r') as file:
                        reader = csv.reader(file)
                        next(reader)
                        inference_times = [float(row[3]) for row in reader]
                    print(f"Finished AD for {len(objects)} objects (seed {seed}), mean inference time: {sum(inference_times)/len(inference_times):.5f} s/sample")

                    # evaluate all finished runs and create sample anomaly maps for inspection
                    print(f"=========== Evaluate seed = {seed} ===========")
                    eval_finished_run(args.dataset, 
                                    args.data_root, 
                                    anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                                    output_dir = results_dir,
                                    seed = seed,
                                    pro_integration_limit = 0.3,
                                    eval_clf = args.eval_clf,
                                    eval_segm = args.eval_segm)

                # delete after generating metrics
                if delete_anomlay_maps:
                    os.system(f"rm -r {results_dir}/anomaly_maps/seed={seed}")

    print("Finished and evaluated all runs!")