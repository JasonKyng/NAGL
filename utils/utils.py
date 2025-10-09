import torch
import random
import numpy as np
import cv2
from torchvision import transforms

from scipy.ndimage import gaussian_filter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_info(dataset):
    
    if dataset == "MVTec":
        objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
        object_anomalies = {"bottle": ["broken_large", "broken_small", "contamination"],
                            "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_wire", "missing_cable", "poke_insulation"],
                            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
                            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
                            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
                            "hazelnut": ["crack", "cut", "hole", "print"],
                            "leather": ["color", "cut", "fold", "glue", "poke"],
                            "metal_nut": ["bent", "color", "flip", "scratch"],
                            "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
                            "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
                            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
                            "toothbrush": ["defective"], 
                            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
                            "wood": ["color", "combined", "hole", "liquid", "scratch"],
                            "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]
                            }

    elif dataset == "VisA":
        objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        object_anomalies = {"candle": ["bad"],
                            "capsules": ["bad"],
                            "cashew": ["bad"],
                            "chewinggum": ["bad"],
                            "fryum": ["bad"],
                            "macaroni1": ["bad"],
                            "macaroni2": ["bad"],
                            "pcb1": ["bad"],
                            "pcb2": ["bad"],
                            "pcb3": ["bad"],
                            "pcb4": ["bad"],
                            "pipe_fryum": ["bad"],
                            }


    elif dataset == "BTAD":
        objects = ["01","02","03"]
        object_anomalies = {"01": ["ko"],
                            "02": ["ko"],
                            "03": ["ko"],
                            }

    
    elif dataset == "BraTS":
        objects = ["brain"]
        object_anomalies = {"brain": ["lesion"],
                            }
    else:
        raise ValueError(f"Dataset '{dataset}' not yet covered!")


    return objects, object_anomalies

def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def get_transform(image_size):
    return transforms.Compose([transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                    [0.229, 0.224, 0.225])
                                ])