"""
re_code.py - Cải tiến Uncertainty-Aware Mean Teacher cho ACDC Dataset

Tác giả: [Your Name]
Mô tả: Cải tiến phương pháp UAMT với:
    1. Loss functions nâng cao (Boundary Loss, Focal Loss, Tversky Loss)
    2. Model cải tiến với Attention và Deep Supervision
    
Dữ liệu ACDC:
    - 4 classes: Background, RV (Right Ventricle), MYO (Myocardium), LV (Left Ventricle)
    - Vấn đề: Class imbalance (background chiếm đa số), ranh giới mờ giữa MYO và LV/RV

Các cải tiến đề xuất:
    - Boundary-aware loss: Cải thiện segmentation tại biên
    - Attention mechanism: Tập trung vào vùng tim
    - Deep supervision: Cải thiện gradient flow
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Improved_UAMT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# ========== CẢI TIẾN: Thêm arguments cho các loss mới ==========
parser.add_argument('--use_boundary_loss', type=int, default=1,
                    help='Sử dụng Boundary Loss để cải thiện biên (0/1)')
parser.add_argument('--use_focal_loss', type=int, default=1,
                    help='Sử dụng Focal Loss cho class imbalance (0/1)')
parser.add_argument('--use_tversky_loss', type=int, default=0,
                    help='Sử dụng Tversky Loss thay Dice (0/1)')
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='Gamma parameter cho Focal Loss')
parser.add_argument('--tversky_alpha', type=float, default=0.7,
                    help='Alpha parameter cho Tversky Loss (FN weight)')
parser.add_argument('--boundary_weight', type=float, default=1.0,
                    help='Trọng số cho Boundary Loss')

args = parser.parse_args()


# =====================================================================
#                    CẢI TIẾN 1: LOSS FUNCTIONS MỚI
# =====================================================================

class BoundaryLoss(nn.Module):
    """
    Optimized Boundary Loss - Sử dụng edge detection thay vì distance transform
    
    Ý tưởng: Sử dụng Sobel filter để detect edges (chạy trên GPU, nhanh hơn
    distance transform trên CPU nhiều lần).
    
    Phù hợp cho ACDC: Ranh giới giữa MYO-LV và MYO-RV thường mờ và khó segment.
    
    Paper: "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
    """
    def __init__(self, n_classes):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes
        # Sobel kernels cho edge detection (GPU-friendly)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _get_boundary_mask(self, labels):
        """Tính boundary mask bằng Sobel filter (nhanh, chạy trên GPU)"""
        # One-hot encode
        labels_one_hot = F.one_hot(labels.long(), self.n_classes).permute(0, 3, 1, 2).float()
        
        # FIX: Đảm bảo Sobel kernels ở cùng device với input
        sobel_x = self.sobel_x.to(labels.device)
        sobel_y = self.sobel_y.to(labels.device)
        
        boundary_masks = []
        for c in range(self.n_classes):
            class_mask = labels_one_hot[:, c:c+1, :, :]
            edge_x = F.conv2d(class_mask, sobel_x, padding=1)
            edge_y = F.conv2d(class_mask, sobel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
            boundary_masks.append(edge)
        
        return torch.cat(boundary_masks, dim=1)
    
    def forward(self, outputs_soft, labels):
        """
        outputs_soft: [B, C, H, W] - softmax predictions
        labels: [B, H, W] - ground truth labels
        """
        boundary_mask = self._get_boundary_mask(labels)
        # Tăng weight cho predictions gần boundary (1-3x weight)
        boundary_weight = 1.0 + 2.0 * boundary_mask
        
        # Weighted cross-entropy style loss
        labels_one_hot = F.one_hot(labels.long(), self.n_classes).permute(0, 3, 1, 2).float()
        boundary_loss = -torch.mean(boundary_weight * labels_one_hot * torch.log(outputs_soft + 1e-6))
        
        return boundary_loss


class FocalLoss(nn.Module):
    """
    Focal Loss - Xử lý class imbalance
    
    Ý tưởng: Giảm weight cho các samples dễ classify (high confidence),
    tăng weight cho samples khó (low confidence).
    
    Formula: FL = -alpha * (1-p)^gamma * log(p)
    
    Phù hợp cho ACDC: Background chiếm đa số pixel, cần focus vào 
    các vùng tim (RV, MYO, LV) khó hơn.
    
    Paper: "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, n_classes, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        
        # Alpha: class weights để cân bằng
        # ACDC specific: Background (0.1), RV (0.3), MYO (0.3), LV (0.3)
        if alpha is None:
            self.alpha = torch.tensor([0.1, 0.3, 0.3, 0.3])
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] - raw logits
        targets: [B, H, W] - ground truth labels
        """
        # Tính CE loss cho mỗi pixel
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        
        # Tính pt (probability của class đúng)
        pt = torch.exp(-ce_loss)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight cho mỗi class
        alpha_weight = self.alpha.to(inputs.device)[targets.long()]
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Điều chỉnh FP/FN trade-off
    
    Ý tưởng: Mở rộng Dice Loss với 2 parameters alpha và beta
    để kiểm soát weight cho False Positives và False Negatives.
    
    Formula: TI = TP / (TP + alpha*FN + beta*FP)
    
    Khi alpha > beta: Ưu tiên giảm FN (recall cao hơn)
    Khi alpha < beta: Ưu tiên giảm FP (precision cao hơn)
    Khi alpha = beta = 0.5: Tương đương Dice Loss
    
    Phù hợp cho ACDC: Với medical imaging, thường ưu tiên
    không bỏ sót vùng bệnh (giảm FN).
    
    Paper: "Tversky loss function for image segmentation" (2017)
    """
    def __init__(self, n_classes, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha  # Weight for FN (default: ưu tiên recall)
        self.beta = beta    # Weight for FP
        self.smooth = smooth
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] - softmax predictions
        targets: [B, 1, H, W] - ground truth labels
        """
        targets_one_hot = self._one_hot_encoder(targets.squeeze(1))
        
        # Flatten
        inputs_flat = inputs.view(-1)
        targets_flat = targets_one_hot.view(-1)
        
        # Tính TP, FP, FN
        TP = (inputs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * inputs_flat).sum()
        FN = (targets_flat * (1 - inputs_flat)).sum()
        
        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        return 1 - tversky_index


class CombinedLoss(nn.Module):
    """
    Combined Loss - Kết hợp nhiều loss functions
    
    Strategy: 
    1. CE Loss: Cơ bản cho classification
    2. Dice/Tversky Loss: Cho overlap optimization
    3. Focal Loss (optional): Cho class imbalance
    4. Boundary Loss (optional): Cho edge improvement
    
    Công thức: L = w1*CE + w2*Dice + w3*Focal + w4*Boundary
    
    Với ACDC, recommend:
    - CE + Dice: Baseline
    - + Focal: Khi class imbalance nghiêm trọng
    - + Boundary: Khi cần cải thiện biên
    """
    def __init__(self, n_classes, 
                 use_focal=True, 
                 use_boundary=True,
                 use_tversky=False,
                 focal_gamma=2.0,
                 tversky_alpha=0.7,
                 boundary_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes
        self.use_focal = use_focal
        self.use_boundary = use_boundary
        self.use_tversky = use_tversky
        
        # Standard losses
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = losses.DiceLoss(n_classes)
        
        # Advanced losses
        if use_focal:
            self.focal_loss = FocalLoss(n_classes, gamma=focal_gamma)
        if use_boundary:
            self.boundary_loss = BoundaryLoss(n_classes)
            self.boundary_weight = boundary_weight
        if use_tversky:
            self.tversky_loss = TverskyLoss(n_classes, alpha=tversky_alpha)
    
    def forward(self, outputs, outputs_soft, labels, iter_num=0, max_iterations=30000):
        """
        outputs: [B, C, H, W] - raw logits
        outputs_soft: [B, C, H, W] - softmax predictions
        labels: [B, H, W] - ground truth
        iter_num: current iteration (for dynamic weighting)
        """
        # Chuẩn bị labels
        labels_ce = labels.long()
        labels_dice = labels.unsqueeze(1)
        
        # CE Loss - always use
        loss_ce = self.ce_loss(outputs, labels_ce)
        
        # Dice or Tversky Loss
        if self.use_tversky:
            # Tversky thay thế Dice
            loss_region = self.tversky_loss(outputs_soft, labels_dice)
        else:
            # Standard Dice
            loss_region = self.dice_loss(outputs_soft, labels_dice)
        
        # Base supervised loss
        supervised_loss = 0.5 * (loss_ce + loss_region)
        
        # Focal Loss (thêm vào CE hoặc thay thế)
        if self.use_focal:
            loss_focal = self.focal_loss(outputs, labels_ce)
            # Blend: 70% focal + 30% CE
            loss_ce_combined = 0.7 * loss_focal + 0.3 * loss_ce
            supervised_loss = 0.5 * (loss_ce_combined + loss_region)
        
        # Boundary Loss (ramp up dần)
        if self.use_boundary:
            # Boundary loss quan trọng hơn ở giai đoạn sau training
            boundary_rampup = ramps.sigmoid_rampup(iter_num, max_iterations * 0.3)
            loss_boundary = self.boundary_loss(outputs_soft, labels)
            supervised_loss = supervised_loss + self.boundary_weight * boundary_rampup * loss_boundary
        
        return supervised_loss, loss_ce, loss_region


# =====================================================================
#                    CẢI TIẾN 2: MODEL ARCHITECTURE
# =====================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (phần của CBAM)
    
    Ý tưởng: Học attention weights cho mỗi channel
    để focus vào các feature maps quan trọng.
    
    Process: GAP + GMP -> Shared MLP -> Sigmoid -> Channel weights
    
    FIX: Đảm bảo hidden_dim >= 4 để MLP có thể học được
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # FIX: Đảm bảo hidden dim >= 4 để tránh bottleneck quá nhỏ
        hidden_dim = max(in_channels // reduction_ratio, 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (phần của CBAM)
    
    Ý tưởng: Học attention weights cho mỗi vị trí spatial
    để focus vào các vùng quan trọng (vùng tim vs background).
    
    Process: Channel pooling -> Conv 7x7 -> Sigmoid -> Spatial weights
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate và convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv(combined))
        
        return x * spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Kết hợp Channel Attention và Spatial Attention
    để focus vào "what" và "where" quan trọng.
    
    Phù hợp cho ACDC: Giúp model focus vào vùng tim
    thay vì background.
    
    Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlockWithAttention(nn.Module):
    """
    Convolution Block với CBAM Attention
    
    Cải tiến từ ConvBlock gốc bằng cách thêm attention
    sau mỗi convolution block.
    """
    def __init__(self, in_channels, out_channels, dropout_p, use_attention=True):
        super(ConvBlockWithAttention, self).__init__()
        self.use_attention = use_attention
        
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x):
        x = self.conv_conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class ResidualConvBlock(nn.Module):
    """
    Residual Convolution Block
    
    Cải tiến bằng cách thêm skip connection
    để cải thiện gradient flow.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ResidualConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection với 1x1 conv nếu channel khác
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.activation = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv_block(x)
        out = out + residual
        return self.activation(out)


class ImprovedEncoder(nn.Module):
    """
    Improved Encoder với Attention và Residual connections
    """
    def __init__(self, params):
        super(ImprovedEncoder, self).__init__()
        self.in_chns = params['in_chns']
        self.ft_chns = params['feature_chns']
        self.dropout = params['dropout']
        
        # Sử dụng attention ở bottleneck levels
        self.in_conv = ConvBlockWithAttention(self.in_chns, self.ft_chns[0], 
                                               self.dropout[0], use_attention=False)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlockWithAttention(self.ft_chns[2], self.ft_chns[3], 
                                   self.dropout[3], use_attention=True)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlockWithAttention(self.ft_chns[3], self.ft_chns[4], 
                                   self.dropout[4], use_attention=True)
        )
    
    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class ImprovedDecoder(nn.Module):
    """
    Improved Decoder với Deep Supervision
    
    Deep Supervision: Tạo prediction ở nhiều scales
    để cải thiện gradient flow và regularization.
    """
    def __init__(self, params):
        super(ImprovedDecoder, self).__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']
        self.bilinear = params.get('bilinear', False)
        
        # Up-sampling blocks
        self.up1 = self._make_up_block(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3])
        self.up2 = self._make_up_block(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2])
        self.up3 = self._make_up_block(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1])
        self.up4 = self._make_up_block(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0])
        
        # Final output
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        
        # Deep supervision outputs
        self.ds_out3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=1)
        self.ds_out2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=1)
        self.ds_out1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=1)
    
    def _make_up_block(self, in_ch1, in_ch2, out_ch):
        return nn.ModuleDict({
            'up': nn.ConvTranspose2d(in_ch1, in_ch2, kernel_size=2, stride=2),
            'conv': ResidualConvBlock(in_ch2 * 2, out_ch, 0.0)
        })
    
    def forward(self, features, return_deep_supervision=False):
        x0, x1, x2, x3, x4 = features
        target_shape = x0.shape[2:]
        
        # Level 4 -> 3
        x = self.up1['up'](x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1['conv'](x)
        if return_deep_supervision:
            ds3 = F.interpolate(self.ds_out3(x), size=target_shape, mode='bilinear', align_corners=True)
        
        # Level 3 -> 2
        x = self.up2['up'](x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2['conv'](x)
        if return_deep_supervision:
            ds2 = F.interpolate(self.ds_out2(x), size=target_shape, mode='bilinear', align_corners=True)
        
        # Level 2 -> 1
        x = self.up3['up'](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3['conv'](x)
        if return_deep_supervision:
            ds1 = F.interpolate(self.ds_out1(x), size=target_shape, mode='bilinear', align_corners=True)
        
        # Level 1 -> 0
        x = self.up4['up'](x)
        x = torch.cat([x, x0], dim=1)
        x = self.up4['conv'](x)
        
        output = self.out_conv(x)
        
        if return_deep_supervision:
            return output, ds1, ds2, ds3
        return output


class ImprovedUNet(nn.Module):
    """
    Improved UNet với:
    1. CBAM Attention ở encoder bottleneck
    2. Residual connections
    3. Deep Supervision (optional)
    
    Số params tăng khoảng 5-10% so với UNet gốc
    nhưng performance cải thiện đáng kể.
    """
    def __init__(self, in_chns, class_num, deep_supervision=True):
        super(ImprovedUNet, self).__init__()
        self.deep_supervision = deep_supervision
        
        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False
        }
        
        self.encoder = ImprovedEncoder(params)
        self.decoder = ImprovedDecoder(params)
    
    def forward(self, x):
        features = self.encoder(x)
        
        if self.training and self.deep_supervision:
            output, ds1, ds2, ds3 = self.decoder(features, return_deep_supervision=True)
            return output, ds1, ds2, ds3
        else:
            output = self.decoder(features, return_deep_supervision=False)
            return output


# =====================================================================
#                    TRAINING FUNCTIONS
# =====================================================================

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ema=False, use_improved=True):
    """
    Tạo model với option sử dụng Improved UNet
    """
    if use_improved:
        # Sử dụng Improved UNet với attention và deep supervision
        model = ImprovedUNet(in_chns=1, class_num=args.num_classes, deep_supervision=True)
    else:
        # Sử dụng UNet gốc
        model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    
    if ema:
        for param in model.parameters():
            param.detach_()
    
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # ===== Tạo models =====
    model = create_model(ema=False, use_improved=True)
    ema_model = create_model(ema=True, use_improved=True)
    
    model = model.cuda()
    ema_model = ema_model.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # ===== Data loaders =====
    db_train = BaseDataSets(
        base_dir=args.root_path, 
        split="train", 
        num=None, 
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print(f"Total slices: {total_slices}, Labeled slices: {labeled_slice}")
    
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # ===== Optimizer =====
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # ===== Loss functions =====
    combined_loss = CombinedLoss(
        n_classes=num_classes,
        use_focal=args.use_focal_loss,
        use_boundary=args.use_boundary_loss,
        use_tversky=args.use_tversky_loss,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        boundary_weight=args.boundary_weight
    )
    
    # Standard losses for backward compatibility
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # ===== Training loop =====
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"{len(trainloader)} iterations per epoch")
    logging.info(f"Using Focal Loss: {args.use_focal_loss}, Boundary Loss: {args.use_boundary_loss}")

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            # ===== Student forward pass =====
            model.train()
            
            # Forward with deep supervision
            model_output = model(volume_batch)
            
            if isinstance(model_output, tuple):
                # Deep supervision enabled
                outputs, ds1, ds2, ds3 = model_output
            else:
                outputs = model_output
                ds1 = ds2 = ds3 = None
            
            outputs_soft = torch.softmax(outputs, dim=1)

            # ===== Teacher forward pass (with noise) =====
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                if isinstance(ema_output, tuple):
                    ema_output = ema_output[0]

            # ===== Uncertainty estimation (Monte Carlo Dropout) =====
            T = 8
            _, _, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            
            ema_model.train()  # Enable dropout for uncertainty estimation
            for i in range(T // 2):
                ema_inputs_mc = volume_batch_r + torch.clamp(
                    torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2
                )
                with torch.no_grad():
                    pred = ema_model(ema_inputs_mc)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    preds[2 * stride * i:2 * stride * (i + 1)] = pred
            ema_model.eval()
            
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            # ===== Supervised Loss (với Combined Loss) =====
            supervised_loss, loss_ce, loss_region = combined_loss(
                outputs[:args.labeled_bs],
                outputs_soft[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                iter_num,
                max_iterations
            )
            
            # Deep supervision loss (if available)
            if ds1 is not None:
                ds_loss = 0.0
                for ds_out in [ds1, ds2, ds3]:
                    ds_soft = torch.softmax(ds_out[:args.labeled_bs], dim=1)
                    ds_loss += dice_loss(ds_soft, label_batch[:args.labeled_bs].unsqueeze(1))
                ds_loss = ds_loss / 3.0
                supervised_loss = supervised_loss + 0.4 * ds_loss  # 40% weight for deep supervision

            # ===== Consistency Loss =====
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = losses.softmax_mse_loss(outputs[args.labeled_bs:], ema_output)
            
            # Adaptive threshold based on uncertainty
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            # ===== Total Loss =====
            loss = supervised_loss + consistency_weight * consistency_loss

            # ===== Backpropagation =====
            optimizer.zero_grad()
            loss.backward()
            # FIX: Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA model
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # Learning rate decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # ===== Logging =====
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_region', loss_region, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            
            logging.info(
                f'iteration {iter_num}: loss={loss.item():.4f}, '
                f'ce={loss_ce.item():.4f}, region={loss_region.item():.4f}'
            )

            # Visualization
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                
                pred_vis = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred_vis[1, ...] * 50, iter_num)
                
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # ===== Validation =====
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                
                for sampled_batch in valloader:
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], 
                        model, classes=num_classes
                    )
                    metric_list += np.array(metric_i)
                
                metric_list = metric_list / len(db_val)
                
                for class_i in range(num_classes - 1):
                    writer.add_scalar(f'info/val_{class_i+1}_dice', metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f'info/val_{class_i+1}_hd95', metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, f'iter_{iter_num}_dice_{round(best_performance, 4)}.pth'
                    )
                    save_best = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(f'iteration {iter_num}: mean_dice={performance:.4f}, mean_hd95={mean_hd95:.4f}')
                model.train()

            # Periodic save
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")

            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:
            iterator.close()
            break
    
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"../model/{args.exp}_{args.labeled_num}_labeled/{args.model}"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt", 
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)
