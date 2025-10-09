import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import argparse

from utils.utils import setup_seed, get_transform
from models.model import NAGL
from utils.dataset import FSDataset

# import time

from utils.metrics import FewShotMetric

def run(args, 
        model, 
        dataloader, 
        optimizer=None,  
        training=True):
    
    if training:
        model.train()
    else:
        model.eval()
    products = dataloader.dataset.products
    mean_loss, mean_loss_i, mean_loss_p = 0, 0, 0

    fs_metric = FewShotMetric(products)
    for i, data in enumerate(dataloader):
        query = data['query']
        query_image = query[0].to(args.device)  # [B, 1, C, H, W]
        query_mask = query[1].squeeze(1).to(args.device)   # [B, 1, H, W]
        sample_product = data['sample_product']

        image_level_label = data['image_level_label'][0].to(args.device) # [B]
        
        support_normal = data['support_normal'] # (img: [B, n_shot, C, H, W], mask: [B, n_shot, H, W]) or None
        support_abnormal = data['support_abnormal'] # (img: [B, a_shot, C, H, W], mask: [B, a_shot, H, W]) or None
        
        image_level_logits, pixel_level_logits, loss_i, loss_p = model(args, query_image, query_mask, image_level_label, support_normal, support_abnormal)
        loss = loss_i + loss_p
        
        fs_metric.update(image_level_logits, image_level_label, pixel_level_logits, query_mask, sample_product)

        mean_loss += loss
        mean_loss_i += loss_i
        mean_loss_p += loss_p

        if i % args.print_freq == 0:
            current_iter = i + 1
            print(f'Iter: {i} \t || Total Loss: {mean_loss/current_iter:.4f}, I-Loss: {mean_loss_i/current_iter:.4f}, P-Loss: {mean_loss_p/current_iter:.4f}')

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    mean_i_roc, mean_p_roc = fs_metric.get_scores()
    fs_metric.print_metrics()
    return mean_i_roc, mean_p_roc, mean_loss/len(dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("NAGL", add_help=True)

    parser.add_argument("--data_root", type=str, default="./path/to/dataset/mvtec", help="train dataset path")
    # parser.add_argument("--data_mode", type=str, default='realiad', choices=['realiad', 'mvtec_visa'], help="train dataset mode")
    parser.add_argument("--data_mode", type=str, default='mvtec_visa', choices=['realiad', 'mvtec_visa'], help="train dataset mode")
    parser.add_argument("--fold", type=int, default=0, help="fold") # 0: val on former (mvtec), training on latter (visa), 1: val on latter (visa), training on former (mvtec)
    parser.add_argument("--save_path", type=str, default='output/checkpoint', help='path to save results')
    
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument('--local_rank', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--port', type=str, default='1234', help='number of cpu threads to use during batch generation')
    
    parser.add_argument("--backbone_name", type=str, default='dinov2_vits14', help="the name of encoder")
    parser.add_argument("--num_learnable_proxies", type=int, default=3, help="number of learnable queries")
    parser.add_argument("--n_shot", type=int, default=1, help="number of normal samples")
    parser.add_argument("--a_shot", type=int, default=1, help="number of abnormal samples")

    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    args = parser.parse_args()

    print(args)

    # Set seed
    setup_seed(args.seed)
    if os.path.exists(f'{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_best.pth'):
        print(f"Results for N-Shot = {args.n_shot}, A-Shot = {args.a_shot} already exist. Skipping.")
        exit()

    # Distributed setting
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    args.device = torch.device('cuda', local_rank)

    # Create model
    model = NAGL(args)
    # Device setup
    model.to(args.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Freeze parameters
    freeze_params_key_name = ['vision_encoder', 
                              'mask_downsample']

    trainable_params = []
    total_params_num = 0
    trainable_num = 0
    print(f"Trainable Parameters: ")
    for param_name, param in model.named_parameters():
        if (param_name.split('.')[1] not in freeze_params_key_name):
            print(param_name, param.shape)
            trainable_num += param.numel()
            param.requires_grad_(True)
            trainable_params.append({'params': param})
        else:
            total_params_num += param.numel()
            param.requires_grad_(False)

    print(f'Total params: {(total_params_num+trainable_num)/1e6:.3f}M \nTrainable params: {trainable_num/1e6:.3f} M \n')

    # Create optimizer and lr scheduler
    optimizer = torch.optim.AdamW(trainable_params, 
                                  lr=args.learning_rate, 
                                  betas=(0.9, 0.999)
                                  )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[10, 15], 
                                                     gamma=0.1)
    
    transform = get_transform((args.image_size, args.image_size))

    # Create dataset and dataloader
    train_data = FSDataset(data_root=args.data_root,  
                           data_mode=args.data_mode,
                           fold=args.fold, 
                           split='train', 
                           shot=[args.n_shot, args.a_shot], 
                           transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(train_data, 
                                  batch_size=args.batch_size, 
                                  pin_memory=True, 
                                  num_workers=args.worker, 
                                  sampler=train_sampler)
    
    val_data = FSDataset(data_root=args.data_root,
                           data_mode=args.data_mode,
                           fold=args.fold, 
                           split='eval', 
                           shot=[args.n_shot, args.a_shot], 
                           transform=transform)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    val_dataloader = DataLoader(val_data,
                                batch_size=args.batch_size, 
                                pin_memory=True, 
                                num_workers=args.worker, 
                                sampler=val_sampler)

    # Training and validation
    best_roc = 0
    for epoch in range(args.epoch):
        print(f'Epoch: {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}')
        print(f'----------Train-----------')
        mean_i_roc, mean_p_roc, mean_loss = run(args, 
                        model, 
                        train_dataloader,
                        optimizer, 
                        training=True)
        scheduler.step()
        print(f'Train Results \t || I-AUROC: {mean_i_roc:.4f}, P-AUROC: {mean_p_roc:.4f}, Loss: {mean_loss:.4f}')
        # save the last checkpoint
        torch.save({name: param for name, param in model.named_parameters() if param.requires_grad}, f'{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_last.pth')

        with torch.no_grad():
            print(f'--------Validation--------')
            mean_i_roc, mean_p_roc, mean_loss = run(args, 
                                       model, 
                                       val_dataloader,
                                       training=False)
            print(f'Val Results \t || I-AUROC: {mean_i_roc:.4f}, P-AUROC: {mean_p_roc:.4f}, Loss: {mean_loss:.4f}\n')

            if (epoch >= 5) and (mean_i_roc + mean_p_roc >= best_roc):
                best_epoch = epoch
                best_i_roc = mean_i_roc
                best_p_roc = mean_p_roc
                best_roc = best_i_roc + best_p_roc
                # save the best model
                torch.save({name: param for name, param in model.named_parameters() if param.requires_grad}, f'{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_best.pth')
            
            if epoch < 5:
                print(f'Warmup Epoch {epoch} \n')
            else:
                print(f'Previous Best I-AUROC: {best_i_roc:.4f}({best_epoch}) || Best P-AUROC: {best_p_roc:.4f}({best_epoch}) \n')
    