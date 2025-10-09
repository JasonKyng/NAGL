import torch
import torch.nn as nn

from torchvision.ops import sigmoid_focal_loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(0)
        negative = negative.unsqueeze(0)

        pos_distance = torch.sum((anchor - positive).pow(2), dim=1)

        neg_distance = torch.sum((anchor - negative).pow(2), dim=1)

        loss = torch.relu(pos_distance - neg_distance + self.margin)

        return torch.mean(loss)

class ContrasitveLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(ContrasitveLoss, self).__init__()
        self.temperature = temperature
    
    def cosine_similarity(self, x, y):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-7)
        return x @ y.t()
    
    def forward(self, pair_p, pair_s, unpair_p):
        # exp_pair = torch.exp(torch.einsum('nc,bc->bn', pair_p, pair_s))/self.temperature
        # exp_unpair = torch.exp(torch.einsum('nc,bc->bn', unpair_p, pair_s))/self.temperature
        exp_pair = torch.exp(torch.einsum('bnc,bnc->bn', pair_p, pair_s))/self.temperature
        exp_unpair = torch.exp(torch.einsum('bnc,bnc->bn', unpair_p, pair_s))/self.temperature
        loss = -torch.log(exp_pair / (exp_pair + exp_unpair)).mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), "input and target must have the same shape"
        return sigmoid_focal_loss(inputs, target, self.alpha, self.gamma, reduction='mean')

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.b_dice = BinaryDiceLoss(smooth)

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size(), "inputs and targets must have the same shape"
        N = targets.size(1)
        if inputs.dim() == 3:
            loss = self.b_dice(inputs, targets)
        else:
            log_inputs = inputs.log_softmax(dim=1).exp()
            loss = 0
            for i in range(N):
                loss += self.b_dice(log_inputs[:, i, :, :], targets[:, i, :, :])
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        N = targets.size()[0]
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + self.smooth) / (input_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss

def smooth(arr, lamda1):
    new_array = arr
    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss

def sparsity(arr, target, lamda2):
    if target == 0:
        loss = torch.mean(torch.norm(arr, dim=0))
    else:
        loss = torch.mean(torch.norm(1-arr, dim=0))
    return lamda2 * loss

class OrthogonalLoss(nn.Module):
    def __init__(self, lambda_ortho=1.0):
        super(OrthogonalLoss, self).__init__()
        self.lambda_ortho = lambda_ortho

    def forward(self, features):
        # features: (batch_size, num_features, num_channels)
        
        # transpose: (batch_size, num_channels, num_features)
        features_t = features.transpose(1, 2)
        
        # Gram matrix: (batch_size, num_channels, num_channels)
        gram_matrix = torch.bmm(features_t, features)
        
        # Identity matrix: (batch_size, num_channels, num_channels)
        identity_matrix = torch.eye(gram_matrix.size(1)).to(features.device)
        identity_matrix = identity_matrix.unsqueeze(0).expand(features.size(0), -1, -1)
        
        # Calculate orthogonal loss for all batches at once
        ortho_loss = torch.norm(gram_matrix - identity_matrix, p='fro', dim=(1,2))
        
        # Update lambda_ortho based on matrix size
        self.lambda_ortho *= 1/gram_matrix.size(1)
        
        # Average loss across batch
        avg_ortho_loss = ortho_loss.mean()
        
        return self.lambda_ortho * avg_ortho_loss
