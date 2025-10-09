r""" Few-Shot Anomaly Detection Dataset """

import os
import json
import random   

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import PIL.Image as Image
import numpy as np

class FSDataset(Dataset):
    def __init__(self,
                 data_root:str ='/data/datasets', 
                 data_mode:str = 'mvtec_visa',
                 data_name_json:str ='meta.json',
                 fold:int =0,
                 split:str ='train',
                 shot:list =[5, 1],
                 transform:object =None, 
                 choice=500):
        self.split = 'val' if split in ['eval', 'test'] else 'train'
        self.data_root = data_root
        data = ['mvtec', 'visa'] if data_mode == 'mvtec_visa' else ['Real-IAD/realiad_1024_unzip']
        # data = ['mvtec', 'visa'] if data_mode == 'mvtec_visa' else ['realiad']
        self.data_mode = data_mode
        self.data_name_json = data_name_json

        self.choice = choice
        # self.nfolds = 2
        self.fold = fold
        self.n_shot, self.a_shot = shot

        self.transform = transform

        self.initialize(data)

        self.class_ids = self.build_class_ids()
 
    def initialize(self, data):
        # Generate metadata
        self.metadata = {}
        self.all_product = []
        for d in data:
            data_path = os.path.join(self.data_root, d)
            with open(os.path.join(data_path, self.data_name_json)) as f:
                # data_info.update(json.load(f)['test'])
                data_info = json.load(f)

            data_info = data_info['test']
            products = list(data_info.keys())
            products.sort()
            self.all_product += products

            for product in products:
                self.metadata.setdefault(product, {'normal': {}, 'abnormal': {}})
                
                for sample in data_info[product]:
                    sample_info = {
                        'img_path': os.path.join(data_path, sample['img_path']),
                        'mask_path': os.path.join(data_path, sample['mask_path']),
                        'anomaly': sample['anomaly']
                    }
                    
                    if sample['anomaly']:
                        product_type = 'abnormal'
                        if sample['specie_name'] == '':
                            sample['specie_name'] = 'bad'
                    else:
                        product_type = 'normal'
                        if sample['specie_name'] == '':
                            sample['specie_name'] = 'good'
                    self.metadata[product][product_type].setdefault(sample['specie_name'], []).append(sample_info)
        
        self.nclass = len(self.all_product)

    def __len__(self):
        # return self.choice
        return self.choice if self.split == 'train' else 2000
    
    def mask_list_transform(self, mask_list, img_size):
        mask_tensor = []
        for mask in mask_list:
            mask_tensor.append(F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), img_size, mode='nearest').squeeze())
        return torch.stack(mask_tensor)
    
    def image_list_transform(self, image_list):
        image_tensor = []
        for image in image_list:
            image_tensor.append(self.transform(image))
        return torch.stack(image_tensor)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # tuple: (image_list, mask_list, anomaly_list)
        query_tuple, support_normal_tuple, support_abnormal_tuple, sample_product = self.load_frame()

        query_image = self.image_list_transform(query_tuple[0]) # [1, C, H, W]
        query_mask = self.mask_list_transform(query_tuple[1], query_image.shape[-2:]) # [1, H, W]
        query_data = [query_image, query_mask]

        support_normal_data = [0, 0]
        support_abnormal_data = [0, 0]
        
        if self.n_shot:
            support_normal_imgs = self.image_list_transform(support_normal_tuple[0])
            support_normal_masks = self.mask_list_transform(support_normal_tuple[1], support_normal_imgs.size()[-2:])
            support_normal_data = [support_normal_imgs, support_normal_masks]
        
        if self.a_shot:
            support_abnormal_imgs = self.image_list_transform(support_abnormal_tuple[0])
            support_abnormal_masks = self.mask_list_transform(support_abnormal_tuple[1], support_abnormal_imgs.size()[-2:])
            support_abnormal_data = [support_abnormal_imgs, support_abnormal_masks]

        ret_dict = {
            'query': query_data,
            'image_level_label': query_tuple[2],
            'support_normal': support_normal_data,
            'support_abnormal': support_abnormal_data,
            'sample_product': sample_product
        }
        
        return ret_dict

    def build_class_ids(self):
        if self.data_mode == 'mvtec_visa': # mvtec and visa
            class_ids_val = range(0, 15) if self.fold == 0 else range(15, 27)
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else: # RealIAD
            self.nfolds = 2
            # number of product categories
            nclass_trn = self.nclass // self.nfolds
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        class_trn = [self.all_product[i] for i in class_ids_trn]
        class_val = [self.all_product[i] for i in class_ids_val]
        class_ids = class_ids_trn if self.split == 'train' else class_ids_val

        msg = f'Train classes: {class_trn}' if self.split == 'train' else  f'Val classes: {class_val} \n'
        print(msg)

        self.products = class_trn if self.split == 'train' else class_val

        return class_ids
    
    def random_sample(self, sample_list, num):
        selected_idx = random.sample(range(len(sample_list)), num)
        selected_sample = [sample_list[i] for i in selected_idx]
        return selected_idx, selected_sample
    
    def get_sample_info(self, sample_dict):
        img_path = sample_dict['img_path']
        mask_path = sample_dict['mask_path']
        anomaly = sample_dict['anomaly']
        return img_path, mask_path, anomaly
    
    def read_mask(self, anomaly, mask_path, imsize):
        if not anomaly:
            mask = torch.zeros(imsize)
        else:
            mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
            mask[mask>0.5] = 1
        return mask
    
    def read_data(self, data_list):
        image_list = []
        mask_list = []
        anomaly_list = []
        for data in data_list:
            data_path, mask_path, anomaly = self.get_sample_info(data)
            image_list.append(Image.open(data_path).convert('RGB'))
            mask_list.append(self.read_mask(anomaly, mask_path, image_list[-1].size))
            anomaly_list.append(anomaly)
        return image_list, mask_list, anomaly_list

    def load_frame(self):

        sample_product_idx = np.random.choice(self.class_ids, 1)[0]
        sample_product = self.all_product[sample_product_idx]

        # sample normal support
        support_normal_specie_type = np.random.choice(list(self.metadata[sample_product]['normal'].keys()), 1)[0]
        support_normal_idx, support_normal = self.random_sample(self.metadata[sample_product]['normal'][support_normal_specie_type], self.n_shot)
        # sample abnormal support
        support_abnormal_specie_type = np.random.choice(list(self.metadata[sample_product]['abnormal'].keys()), 1)[0]
        if len(self.metadata[sample_product]['abnormal'][support_abnormal_specie_type])<self.a_shot:
            print(self.metadata[sample_product]['abnormal'][support_abnormal_specie_type])
        support_abnormal_idx, support_abnormal = self.random_sample(self.metadata[sample_product]['abnormal'][support_abnormal_specie_type], self.a_shot)

        # sample query image
        query_type = np.random.choice(['normal', 'abnormal'], 1)[0]
        query_specie_type = support_normal_specie_type if query_type == 'normal' else support_abnormal_specie_type
        query_idx, query_data = self.random_sample(self.metadata[sample_product][query_type][query_specie_type], 1)
        
        # check if query image is in support set
        if ((query_idx in support_normal_idx and query_type == 'normal') or 
            (query_idx in support_abnormal_idx and query_type == 'abnormal')):
            self.load_frame()

        # Get query image
        query_tuple = self.read_data(query_data)
        
        # Get normal support images
        support_normal_tuple = (0, 0, 0)
        if self.n_shot > 0:
            support_normal_tuple = self.read_data(support_normal)

        # Get abnormal support images
        support_abnormal_tuple = (0, 0, 0)
        if self.a_shot > 0:
            support_abnormal_tuple = self.read_data(support_abnormal)

        return query_tuple, support_normal_tuple, support_abnormal_tuple, sample_product