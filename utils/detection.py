import os 
import time
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm
import tifffile as tiff

from utils.utils import dists2map, get_transform
from utils.post_eval import mean_top1p

# NORMAL_NAME = 'ok'
NORMAL_NAME = 'good'

def get_ref_sample_dir(data_root, object_name, mode, type_anomaly, ref_samples, seed):
    img_ref_folder = f"{data_root}/{object_name}/{mode}/{type_anomaly}"
    if ref_samples == -1:
        img_ref_samples = sorted(glob.glob(os.path.join(img_ref_folder, "*")))
    else:
        img_ref_samples = sorted(glob.glob(os.path.join(img_ref_folder, "*")))[seed*ref_samples:(seed + 1)*ref_samples]

    if len(img_ref_samples) < ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
    
    return img_ref_samples

def run_anomaly_detection(
        args,
        model,
        object_name,
        data_root,
        object_anomalies,
        plots_dir,
        seed = 0,
        save_patch_dists = True,
        save_tiffs = False):
    """
    Main function to evaluate the anomaly detection performance of a given object/product.
    """

    device = args.device
    transform = get_transform((args.image_size, args.image_size))

    normal_ref_dir = get_ref_sample_dir(data_root, object_name, 'train', NORMAL_NAME, args.n_shot, seed)
    normal_ref = []
    normal_ref_mask = []
    for ref_dir in normal_ref_dir:
        img_ref, _ = model.prepare_test_image(ref_dir, transform)
        normal_ref.append(img_ref)
        mask_ref = torch.zeros(img_ref.shape[-2:])
        normal_ref_mask.append(mask_ref)
    normal_ref = torch.stack(normal_ref, dim=0).unsqueeze(0).to(device)
    normal_ref_mask = torch.stack(normal_ref_mask, dim=0).unsqueeze(0).to(device)
    support_normal = normal_ref, normal_ref_mask
    # normal_feat = model.feature_forward(torch.stack(normal_ref, dim=0).unsqueeze(0))

    type_anomalies = object_anomalies[object_name].copy()
    original_type_anomalies = type_anomalies.copy()

    type_anomalies.append(NORMAL_NAME)
    # ensure that each type is only evaluated once
    type_anomalies = sorted(list(set(type_anomalies)))
    inference_times = {}
    anomaly_scores = {}

    # Evaluate anomalies for each anomaly type (and NORMAL_NAME)
    for type_anomaly in tqdm(type_anomalies, desc = f"processing test samples ({object_name})"):
        data_dir = f"{data_root}/{object_name}/test/{type_anomaly}"
        
        if save_patch_dists or save_tiffs:
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}", exist_ok=True)
        
        support_abnormal = (None, None)
        if args.a_shot > 0:
            if type_anomaly == NORMAL_NAME:
                abnormal_ref_dir = get_ref_sample_dir(data_root, object_name, 'test', np.random.choice(original_type_anomalies), args.a_shot, seed)
            else:
                abnormal_ref_dir = get_ref_sample_dir(data_root, object_name, 'test', type_anomaly, args.a_shot, seed)
            abnormal_ref = []
            abnormal_ref_mask = []
            for ref_dir in abnormal_ref_dir:
                img_ref, _ =  model.prepare_test_image(ref_dir, transform)
                grid_size1 = img_ref.shape[-2:]
                abnormal_ref.append(img_ref)
                # if "/03/" in ref_dir:
                #     mask_ref = cv2.cvtColor(cv2.imread(ref_dir.replace('test', 'ground_truth'), 
                #             cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)  
                # else:
                    # mask_ref = cv2.cvtColor(cv2.imread(ref_dir.replace('test', 'ground_truth').replace('.bmp','.png'), 
                    #                             cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)                
                mask_ref = cv2.cvtColor(cv2.imread(ref_dir.replace('test', 'ground_truth').replace('.png','_mask.png').replace('.JPG','.png'), 
                                            cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY) # MvTec ViSA
                # mask_ref = cv2.cvtColor(cv2.imread(ref_dir.replace('test', 'ground_truth').replace('.jpg','_mask.png').replace('.JPG','.png'), 
                #                             cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY) # BraTS
                # mask_ref = mask_downsample(mask_ref, (mask_ref.shape[0]//grid_size1[0], mask_ref.shape[1]//grid_size1[1]))
                # if mask_ref.shape != grid_size1:
                mask_ref = cv2.resize(mask_ref, (grid_size1[1], grid_size1[0]), interpolation=cv2.INTER_NEAREST)

                abnormal_ref_mask.append(torch.tensor(mask_ref, dtype=torch.float32))
            
            abnormal_ref = torch.stack(abnormal_ref, dim=0).unsqueeze(0).to(device)
            abnormal_ref_mask = torch.stack(abnormal_ref_mask, dim=0).unsqueeze(0).to(device)
            support_abnormal = (abnormal_ref, abnormal_ref_mask)

        for idx, img_test_nr in enumerate(sorted(os.listdir(data_dir))):
            # start measuring time (inference)
            start_time = time.time()
            image_test_path = f"{data_dir}/{img_test_nr}"

            # Extract test features
            image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_tensor2, grid_size2 = model.prepare_test_image(image_test, transform)
            query_image = image_tensor2.unsqueeze(0).unsqueeze(0).to(device)
            # features2 = model.extract_features(image_tensor2)
            distances = model(args, query_image, None, None, support_normal, support_abnormal, mode='test').cpu().detach().numpy()
            
            output_distances = distances.squeeze()
            d_masked = output_distances.reshape(grid_size2)
            
            # save inference time
            torch.cuda.synchronize() # Synchronize CUDA kernels before measuring time
            inf_time = time.time() - start_time
            inference_times[f"{type_anomaly}/{img_test_nr}"] = inf_time
            anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = mean_top1p(output_distances.flatten())

            # Save the anomaly maps (raw as .npy or full resolution .tiff files)
            img_test_nr = img_test_nr.split(".")[0]
            if save_tiffs:
                anomaly_map = dists2map(d_masked, image_test.shape)
                tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)
            if save_patch_dists:
                np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.npy", d_masked)

    return anomaly_scores, inference_times