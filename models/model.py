import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
from einops import rearrange

from models.attention_layer import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine
from utils.loss import FocalLoss, DiceLoss

class NAGL(nn.Module):
    def __init__(self, 
                 args
                 ):
        super(NAGL, self).__init__()
        self.backbone_name = args.backbone_name

        self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 
                                            args.backbone_name, 
                                            )

        self.hidden_dim = 384 #vits14
        self.d_scale = 14
        self.vision_encoder.eval()
        # conv downsample for mask
        self.mask_downsample = nn.Conv2d(1, 1, 
                                         kernel_size=self.d_scale, 
                                         stride=self.d_scale, 
                                         padding=1, 
                                         bias=False)
        nn.init.constant_(self.mask_downsample.weight, 1.0)

        self.nheads = 4
        self.pre_norm = False
        self.learnable_proxies = nn.Embedding(args.num_learnable_proxies, self.hidden_dim)
        self.rm_ca, self.rm_sa = self.attention_module() # RM Module
        self.afl_ca, self.afl_sa = self.attention_module() # AFL Module
        self.pe_layer = PositionEmbeddingSine(self.hidden_dim//2, normalize=True)

        # losses
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def attention_module(self):
        cross_attention = CrossAttentionLayer(                                                                                                                                             
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dropout=0.0,
                normalize_before=self.pre_norm,
            ) 
        self_attention = SelfAttentionLayer(
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dropout=0.0,
                normalize_before=self.pre_norm,
            )
        return cross_attention, self_attention

    @torch.no_grad()
    def feature_forward(self, image):
        # get images features from vision encoder
        b, num, c, h, w = image.shape
        feat = self.vision_encoder.get_intermediate_layers(image.view(-1, c, h, w))[0]
        feat = feat.view(b, num, -1, self.hidden_dim)
        return feat
    
    def get_mask(self, mask, target_size):
        '''
        mask: (b, num, h, w)
        '''
        b, _, _, _ = mask.shape
        mask = rearrange(mask, 'b num h w -> (b num) h w').unsqueeze(1)
        mask = self.mask_downsample(mask)
        mask[mask>=1] = 1
        if mask.shape[-2] != target_size:
            mask = F.interpolate(mask, size=(target_size, target_size), mode='nearest', align_corners=True)
        mask = rearrange(mask, '(b num) 1 p_h p_w -> b num (p_h p_w)', b=b).unsqueeze(-1)
        return mask
    
    def nn_search(self, query_feat, support_feat, support_mask=None, mode='max'):
        '''
        query_feat: (b, num_q, M, c)
        support_feat: (b, num_s, N, c)
        support_mask: (b, num_s, h, w)
        '''
        _, num_q, M, _ = query_feat.shape
        _, num_s, N, _ = support_feat.shape
        query_feat = rearrange(F.normalize(query_feat, dim=-1), 'b num_q M c -> b (num_q M) c')
        support_feat = rearrange(F.normalize(support_feat, dim=-1), 'b num_s N c -> b (num_s N) c')
        # query_feat = rearrange(query_feat, 'b 1 M c -> b M c')
        # support_feat = rearrange(support_feat, 'b num_s N c -> b (num_s N) c')
        if support_mask is not None:
            support_mask_ = self.get_mask(support_mask, int(N**0.5))
            support_mask = rearrange(support_mask_, 'b num_s N 1 -> b 1 (num_s N)').repeat(1, M, 1)
        else:
            support_mask_ = None
            support_mask = 1

        similarity_map = (1+torch.einsum('bmc,bnc->bmn', query_feat, support_feat))/2*support_mask
        if mode == 'max':
            pseudo_mask = similarity_map.max(dim=-1)[0]
        elif mode == 'mean':
            pseudo_mask = similarity_map.mean(dim=-1)
        pseudo_mask = rearrange(pseudo_mask, 'b (num_q M) -> b num_q M 1', num_q=num_q)

        return pseudo_mask, support_mask_

    def attention_forward(self, cross_layer, self_layer, query_embed, key_feat, value_feat, feat_mask=None):
        '''
        query_embed: (num_q, c)
        key_feat: (b, num_s, M, c)
        value_feat: (b, num_s, M, c)
        feat_mask: (b, num_s, M, 1)
        '''
        B, _, _, C = value_feat.shape

        if isinstance(query_embed, nn.Embedding):
            # gaussian init
            # nn.init.normal_(query_embed.weight, mean=0, std=0.02)
            q_supp_out = query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        elif query_embed.dim() == 2:
            q_supp_out = query_embed.unsqueeze(1).repeat(1, B, 1)
        else:
            q_supp_out = query_embed.permute(1, 0, 2)

        key = rearrange(key_feat, 'b num M c -> (num M) b c')
        pos_embedding = self.pe_layer(value_feat, feat_mask)
        value = rearrange(value_feat, 'b num M c -> (num M) b c')

        if feat_mask is not None:
            attn_mask = rearrange(feat_mask.squeeze(-1), 'b num M -> b (num M)').unsqueeze(1).repeat(self.nheads, q_supp_out.shape[0], 1)
            attn_mask = -1e9*(1-attn_mask)
        else:
            attn_mask = feat_mask

        output = cross_layer(q_supp_out, key, value, 
                             memory_mask=attn_mask, 
                             memory_key_padding_mask=None,
                             pos=pos_embedding, query_pos=None
                             )
        output = self_layer(output, 
                            tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                            )

        return output.permute(1, 0, 2)

    def get_res_feat(self, ori_feat, temp_feat):
        '''
        ori_feat: (b, num, h*w, c)
        temp_feat: (b, num, h*w, c)
        '''
        b, num, h_w, c = ori_feat.shape
        ori_feat = rearrange(ori_feat, 'b num h_w c -> b (num h_w) c')
        temp_feat = rearrange(temp_feat, 'b num h_w c -> b (num h_w) c')
        sim_map = (1+torch.einsum('bmc,bnc->bmn', ori_feat, temp_feat))/2
        max_idx = sim_map.max(dim=-1)[1]
        most_sim_feat = torch.gather(temp_feat, 1, max_idx.unsqueeze(-1).repeat(1, 1, c))
        res_feat = ori_feat - most_sim_feat
        res_feat = rearrange(res_feat, 'b (num h_w) c -> b num h_w c', b=b, num=num)
        return res_feat
    
    def prepare_test_image(self, img, transform):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.vision_encoder.patch_size, height - height % self.vision_encoder.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.vision_encoder.patch_size, cropped_width // self.vision_encoder.patch_size)
        # return image_tensor
        return image_tensor, grid_size
    
    def forward(self, args, query_image, query_mask, query_label, support_normal, support_abnormal, mode='train'):

        query_feat = self.feature_forward(query_image)

        loss_i = 0
        loss_p = 0

        if args.n_shot>0:
            support_n_image, support_n_mask_ = support_normal
            support_n_feat = self.feature_forward(support_n_image)
            n_pseudo_mask, support_n_mask = self.nn_search(query_feat, support_n_feat, 1-support_n_mask_)
            s_n = n_pseudo_mask.squeeze(-1)
        
        if args.a_shot>0:
            support_a_image, support_a_mask_ = support_abnormal
            support_a_feat = self.feature_forward(support_a_image)
            _, support_a_mask = self.nn_search(query_feat, support_a_feat, support_a_mask_)

            if args.n_shot==0: # only abnormal as reference, abtain normal reference from abnormal image
                support_n_feat = support_a_feat.masked_select(support_a_mask.bool()).view(support_a_feat.shape[0], 1, -1, support_a_feat.shape[-1]) 
                n_pseudo_mask, support_n_mask = self.nn_search(query_feat, support_a_feat, 1-support_a_mask)
                s_n = n_pseudo_mask.squeeze(-1)

            # RM Module forward
            support_res_feat = self.get_res_feat(support_a_feat, support_n_feat)
            residual_proxies = self.attention_forward(self.rm_ca, self.rm_sa, self.learnable_proxies, support_a_feat, support_res_feat, support_a_mask)

            query_res_feat = self.get_res_feat(query_feat, support_n_feat)
            anomaly_proxies = self.attention_forward(self.afl_ca, self.afl_sa, residual_proxies, query_res_feat, query_feat, None).unsqueeze(1)
            
            a_out, _ = self.nn_search(query_feat, anomaly_proxies, mode='mean')
            s_a = a_out.squeeze(-1)
        
        if args.n_shot>0 and args.a_shot==0:
            s_a = 1-s_n
        elif args.n_shot==0 and args.a_shot>0:
            s_n = 1-s_a
        else:
            assert args.n_shot>0 or args.a_shot>0, 'n_shot and a_shot should not be both 0'

        pixel_level_logits = torch.cat([s_n, s_a], dim=1) # (b, 2, h*w)
        
        a_score = (s_a+(1-s_n))/2

        if mode == 'train':
            a_score_topk = torch.topk(a_score, 20, dim=-1)[0].mean(dim=-1)
            image_level_logits = torch.cat([1-a_score_topk, a_score_topk], dim=-1)

            # Image Level
            loss_i += self.cross_entropy_loss(image_level_logits, query_label.long())

            # Pixel Level
            l = int(pixel_level_logits.shape[-1]**0.5)
            pixel_level_logits = rearrange(pixel_level_logits, 'b n (h w) -> b n h w', h=l)
            pixel_level_logits = F.interpolate(pixel_level_logits, size=query_mask.shape[-2:], mode='bilinear')
            query_mask_n = torch.stack([1-query_mask, query_mask], dim=1)
            loss_p += self.focal_loss(pixel_level_logits, query_mask_n)
            loss_p += self.dice_loss(pixel_level_logits, query_mask_n)

            return image_level_logits, pixel_level_logits, loss_i, loss_p
        
        elif mode == 'test':
            return a_score
