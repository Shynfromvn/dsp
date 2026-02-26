"""
re_code_v3.py - Cải tiến UAMT V3 cho ACDC Dataset

Cải tiến so với V1 (re_code.py):

=== LOSS FUNCTIONS ===
1. Exponential Logarithmic Loss (thay CE + Focal): 
   - Ổn định gradient hơn, xử lý class imbalance tự động
   - Chỉ 1 loss thay vì 2 conflicting losses
2. Class-Adaptive Dice Loss (thay DiceLoss gốc):
   - Weight per-class tự động theo inverse frequency
   - RV/MYO/LV nhỏ → weight cao hơn background tự động
3. Hausdorff Distance Loss (thay BoundaryLoss sai):
   - Xấp xỉ distance transform differentiable
   - Tối ưu trực tiếp metric HD95 dùng để đánh giá
4. KL Divergence cho consistency (thay MSE):
   - Phù hợp hơn cho so sánh probability distributions

=== MODEL ===
5. Attention Gate tại skip connections (thay CBAM ở encoder):
   - Decoder "nói" cho encoder biết cần gì → filter noise
6. SE Blocks nhẹ hơn CBAM (giảm overfitting):
   - Channel attention đủ tốt, bỏ spatial attention
7. ASPP tại bottleneck:
   - Multi-scale context cho cấu trúc tim ở nhiều kích thước
8. Wider channels [32, 64, 128, 256, 512]:
   - Tăng capacity cho attention có đủ diversity
9. GroupNorm thay BatchNorm:
   - Ổn định hơn với small batch (labeled_bs=8-12)
10. Giảm dropout [0.05, 0.1, 0.15, 0.2, 0.3]:
    - Tránh over-regularization với ít labeled data

=== TRAINING ===
11. AdamW + Cosine Annealing LR + Warmup
12. EMA decay scheduling 0.99 → 0.999
13. Deep supervision weight giảm dần
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
                    default='ACDC/Improved_UAMT_V3', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu — giảm từ 24 còn 16 vì model lớn hơn')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# V3 params
parser.add_argument('--warmup_iters', type=int, default=1000,
                    help='Số iterations warmup')
parser.add_argument('--hd_loss_weight', type=float, default=0.5,
                    help='Trọng số Hausdorff Distance Loss')
parser.add_argument('--el_loss_gamma', type=float, default=0.3,
                    help='Gamma cho Exponential Logarithmic Loss')
parser.add_argument('--kl_consistency_weight', type=float, default=0.5,
                    help='Tỷ lệ KL trong mixed consistency')

args = parser.parse_args()


# =====================================================================
#              CẢI TIẾN 1: LOSS FUNCTIONS (sửa tất cả vấn đề V1)
# =====================================================================

class ExponentialLogarithmicLoss(nn.Module):
    """
    THAY THẾ: CE + Focal Loss trong V1 (khắc phục conflict + redundancy)
    
    Paper: "3D Segmentation with Exponential Logarithmic Loss" (MICCAI 2018)
    
    Ý tưởng chính:
    - CE và Focal đều optimize cross-entropy → conflict khi dùng cùng lúc
    - EL Loss KẾT HỢP Dice + CE trong KHÔNG GIAN LOGARITHMIC:
      L = w_dice * mean(-ln(Dice_c))^γ  +  w_ce * mean((-ln(p_t))^γ)
    
    Tại sao tốt hơn V1:
    - γ < 1 tự động focus vào classes khó (như Focal) mà KHÔNG cần alpha weights
    - Log transform smooth gradient → ổn định training hơn
    - 1 loss thay 2 losses → không gradient conflict
    
    ACDC: Background lớn (γ < 1 tự động giảm weight), RV/MYO/LV nhỏ → được focus
    """
    def __init__(self, n_classes, w_dice=0.8, w_ce=0.2, gamma=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.w_dice = w_dice  # Dice component weight (chính, cho overlap)
        self.w_ce = w_ce      # CE component weight (phụ, cho pixel-wise)
        self.gamma = gamma    # < 1: ưu tiên classes khó; > 1: ưu tiên classes dễ
    
    def _dice_per_class(self, pred, target):
        """Tính Dice score RIÊNG TỪNG CLASS (khắc phục lỗi flatten V1)"""
        smooth = 1e-5
        dice_scores = []
        for c in range(self.n_classes):
            pred_c = pred[:, c]                    # [B, H, W]
            target_c = (target == c).float()       # [B, H, W]
            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            dice_scores.append(dice)
        return dice_scores
    
    def forward(self, outputs_soft, labels):
        """
        outputs_soft: [B, C, H, W] - softmax predictions
        labels: [B, H, W] - ground truth labels
        """
        # Dice component: -ln(Dice_c) rồi power gamma, rồi trung bình
        dice_scores = self._dice_per_class(outputs_soft, labels)
        dice_log = torch.stack([-torch.log(d + 1e-7) for d in dice_scores])
        loss_dice = torch.mean(dice_log ** self.gamma)
        
        # CE component: (-ln(p_t))^gamma cho mỗi pixel
        labels_long = labels.long()
        pt = outputs_soft.gather(1, labels_long.unsqueeze(1)).squeeze(1)  # p(đúng class)
        loss_ce = torch.mean((-torch.log(pt + 1e-7)) ** self.gamma)
        
        return self.w_dice * loss_dice + self.w_ce * loss_ce


class ClassAdaptiveDiceLoss(nn.Module):
    """
    THAY THẾ: DiceLoss gốc (losses.DiceLoss) cho supervised loss
    
    Vấn đề V1 DiceLoss:
    - Tính Dice mỗi class rồi chia đều: loss /= n_classes
    - Background (chiếm 70%+ pixels) và RV (chiếm 5%) có CÙNG weight = 1/4
    - → Model ưu tiên background vì dễ, dice background luôn cao → ít gradient
    
    Cải tiến:
    - Weight = 1/sqrt(class_frequency) tự động trong mỗi batch
    - Class nhỏ (RV/MYO/LV) → frequency thấp → weight CAO hơn
    - Background lớn → frequency cao → weight THẤP hơn
    - Không cần set thủ công alpha như Focal Loss
    """
    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] - softmax predictions
        targets: [B, 1, H, W] hoặc [B, H, W]
        """
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        
        total_loss = 0.0
        total_weight = 0.0
        
        for c in range(self.n_classes):
            pred_c = inputs[:, c]
            target_c = (targets == c).float()
            
            # Dice per class
            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + self.smooth) / \
                   (pred_c.sum() + target_c.sum() + self.smooth)
            
            # Adaptive weight: inverse sqrt frequency
            # Ví dụ: background 70% → weight ≈ 1.2, RV 5% → weight ≈ 4.5
            class_freq = target_c.sum() / (target_c.numel() + self.smooth)
            weight = 1.0 / (torch.sqrt(class_freq) + self.smooth)
            
            total_loss += weight * (1.0 - dice)
            total_weight += weight
        
        return total_loss / (total_weight + self.smooth)


class HausdorffDistanceLoss(nn.Module):
    """
    THAY THẾ: BoundaryLoss V1 (sai implementation)
    
    Vấn đề V1 BoundaryLoss:
    - Dùng Sobel filter → CHỈ detect edge (binary)
    - Không có thông tin KHOẢNG CÁCH từ pixel đến boundary
    - Thực chất V1 là weighted CE near boundary, KHÔNG phải Boundary Loss
    
    Cải tiến:
    - Xấp xỉ distance transform bằng multi-scale erosion (differentiable)
    - Penalize dự đoán sai NẶNG HƠN nếu pixel ở xa boundary
    - Trực tiếp tối ưu metric HD95 được dùng để đánh giá
    
    Paper: "Reducing the Hausdorff Distance" (IEEE TMI 2020)
    """
    def __init__(self, n_classes, alpha=2.0):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        # Multi-scale erosion kernels → xấp xỉ distance levels
        self.register_buffer('kernel_3', self._create_erosion_kernel(3))
        self.register_buffer('kernel_5', self._create_erosion_kernel(5))
        self.register_buffer('kernel_7', self._create_erosion_kernel(7))
    
    def _create_erosion_kernel(self, size):
        """Circular kernel cho morphological erosion"""
        kernel = torch.ones(1, 1, size, size)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 > center**2:
                    kernel[0, 0, i, j] = 0
        return kernel / kernel.sum()
    
    def _approximate_distance_map(self, mask):
        """
        Xấp xỉ distance transform:
        - Erosion 3x3 → pixels ở distance = 1 từ boundary
        - Erosion 5x5 → distance = 2
        - Erosion 7x7 → distance = 3
        → Tạo distance map nhiều level
        """
        eroded_1 = F.conv2d(mask, self.kernel_3.to(mask.device), padding=1)
        boundary_1 = mask - (eroded_1 > 0.5).float() * mask
        
        eroded_2 = F.conv2d(mask, self.kernel_5.to(mask.device), padding=2)
        boundary_2 = mask - (eroded_2 > 0.5).float() * mask - boundary_1
        boundary_2 = torch.clamp(boundary_2, 0, 1)
        
        eroded_3 = F.conv2d(mask, self.kernel_7.to(mask.device), padding=3)
        boundary_3 = mask - (eroded_3 > 0.5).float() * mask - boundary_1 - boundary_2
        boundary_3 = torch.clamp(boundary_3, 0, 1)
        
        dist_map = boundary_1 * 1.0 + boundary_2 * 2.0 + boundary_3 * 3.0
        max_val = dist_map.max() + 1e-7
        return dist_map / max_val
    
    def forward(self, outputs_soft, labels):
        total_loss = 0.0
        n_valid = 0
        
        for c in range(1, self.n_classes):  # Bỏ background
            pred_c = outputs_soft[:, c:c+1]
            target_c = (labels == c).float().unsqueeze(1)
            
            if target_c.sum() < 10:  # Class không tồn tại trong batch
                continue
            
            dist_target = self._approximate_distance_map(target_c)
            dist_pred = self._approximate_distance_map((pred_c > 0.5).float())
            
            # Error × (distance + 1)^alpha → sai xa boundary bị phạt nặng
            error = (pred_c - target_c) ** 2
            weighted_error = error * (dist_target + dist_pred + 1.0) ** self.alpha
            
            total_loss += weighted_error.mean()
            n_valid += 1
        
        if n_valid == 0:
            return torch.tensor(0.0, device=outputs_soft.device, requires_grad=True)
        return total_loss / n_valid


class CombinedLossV3(nn.Module):
    """
    Chiến lược Loss:
    
    Phase 1 (0-40% training): EL Loss + Adaptive Dice
      → Học hình dạng tổng thể, cân bằng classes
    Phase 2 (40-100%): + HD Loss ramp up dần
      → Tinh chỉnh boundary sau khi đã có hình dạng đúng
    
    So sánh V1 → V3:
    ┌─────────────┬────────────────────────┬─────────────────────────┐
    │  Component  │          V1            │           V3            │
    ├─────────────┼────────────────────────┼─────────────────────────┤
    │ Classification │ CE + Focal (conflict)│ EL Loss (unified)      │
    │ Region       │ DiceLoss (equal weight)│ Adaptive Dice (auto)  │
    │ Boundary    │ Sobel-based (sai)      │ HD Loss (distance-aware)│
    │ Tổng losses │ 4 (nhiều conflict)     │ 2-3 (complementary)    │
    └─────────────┴────────────────────────┴─────────────────────────┘
    """
    def __init__(self, n_classes, el_gamma=0.3, hd_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.hd_weight = hd_weight
        
        self.el_loss = ExponentialLogarithmicLoss(n_classes, w_dice=0.8, w_ce=0.2, gamma=el_gamma)
        self.adaptive_dice = ClassAdaptiveDiceLoss(n_classes)
        self.hd_loss = HausdorffDistanceLoss(n_classes)
        
        # Cho logging
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = losses.DiceLoss(n_classes)
    
    def forward(self, outputs, outputs_soft, labels, iter_num=0, max_iterations=30000):
        labels_ce = labels.long()
        labels_dice = labels.unsqueeze(1)
        
        # EL Loss: thay thế CE + Focal
        loss_el = self.el_loss(outputs_soft, labels)
        
        # Adaptive Dice: thay thế DiceLoss equal-weight
        loss_adice = self.adaptive_dice(outputs_soft, labels)
        
        # Kết hợp: 50/50
        supervised_loss = 0.5 * loss_el + 0.5 * loss_adice
        
        # HD Loss: chỉ bật sau 40% training (khi model đã học shape cơ bản)
        hd_rampup_start = max_iterations * 0.4
        if iter_num > hd_rampup_start:
            hd_rampup = ramps.sigmoid_rampup(
                iter_num - hd_rampup_start,
                max_iterations * 0.3
            )
            loss_hd = self.hd_loss(outputs_soft, labels)
            supervised_loss = supervised_loss + self.hd_weight * hd_rampup * loss_hd
        
        # Logging (không backprop)
        with torch.no_grad():
            loss_ce_log = self.ce_loss(outputs, labels_ce)
            loss_dice_log = self.dice_loss(outputs_soft, labels_dice)
        
        return supervised_loss, loss_ce_log, loss_dice_log


# =====================================================================
#              CẢI TIẾN 2: MODEL ARCHITECTURE
# =====================================================================

class SqueezeExcitation(nn.Module):
    """
    THAY THẾ: CBAM (V1) → SE Block (nhẹ hơn ~40%)
    
    Tại sao SE tốt hơn CBAM cho ACDC:
    - CBAM = Channel Attention + Spatial Attention = ~2x params
    - SE = Chỉ Channel Attention = nhẹ hơn, ít overfit hơn
    - Với labeled data ít (7 bệnh nhân), model nhẹ hơn = generalizable hơn
    - SE đã chứng minh hiệu quả tương đương CBAM trong medical imaging
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionGate(nn.Module):
    """
    MỚI: Attention Gate tại skip connections
    
    Paper: "Attention U-Net" (MIDL 2018)
    
    Vấn đề V1: CBAM ở encoder refine features TRƯỚC skip connection
    → Decoder nhận features đã refined nhưng KHÔNG biết cần gì
    
    AttentionGate: Decoder gửi "gating signal" nói cho encoder features 
    biết phần NÀO quan trọng → lọc bỏ background noise, giữ cardiac features
    
    Cách hoạt động:
    1. Decoder feature (g) → 1x1 conv → hiểu "decoder đang cần gì"
    2. Encoder feature (x) → 1x1 conv → hiểu "encoder có gì"
    3. Cộng 2 signals → ReLU → 1x1 conv → Sigmoid → attention map
    4. attention_map × encoder_features → features đã filtered
    """
    def __init__(self, F_g, F_l, F_int):
        """F_g: decoder channels, F_l: encoder channels, F_int: intermediate"""
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """g: gating (decoder), x: encoder features"""
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(self.W_g(g_up) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    """
    MỚI: Atrous Spatial Pyramid Pooling tại bottleneck
    
    Paper: "DeepLab v3+" (ECCV 2018)
    
    Tại sao cần cho ACDC:
    - Tim có cấu trúc multi-scale: LV cavity (lớn), MYO wall (mỏng), RV (phức tạp)
    - Conv 3x3 thông thường chỉ capture 1 scale
    - Dilated conv rate=6,12 capture context ở scale lớn hơn CÙNG LÚC
    - Global Average Pooling capture toàn bộ context
    → Kết hợp cho model "nhìn" ở nhiều kích thước đồng thời
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # Giảm từ 0.5 V1
        )
    
    def forward(self, x):
        size = x.shape[2:]
        out = torch.cat([
            self.conv1(x), self.conv6(x), self.conv12(x),
            F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        ], dim=1)
        return self.fuse(out)


class ResConvBlock(nn.Module):
    """
    CẢI TIẾN: ResidualConvBlock V1 + SE + GroupNorm
    
    Thay đổi:
    1. BatchNorm → GroupNorm: Ổn định hơn với small batch (labeled_bs=8)
       BatchNorm dùng batch statistics → noise khi batch nhỏ
       GroupNorm dùng channel groups → không phụ thuộc batch size
    2. + SE Block: Channel recalibration nhẹ
    3. Dropout2d thay Dropout: Tốt hơn cho feature maps (drop toàn channel)
    """
    def __init__(self, in_channels, out_channels, dropout_p, use_se=True):
        super().__init__()
        self.use_se = use_se
        num_groups = min(32, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
        )
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
        if use_se:
            self.se = SqueezeExcitation(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv_block(x)
        if self.use_se:
            out = self.se(out)
        return self.activation(out + residual)


class EncoderV3(nn.Module):
    """
    Encoder với wider channels + SE blocks ở mọi level
    
    Channels: [16,32,64,128,256] V1 → [32,64,128,256,512] V3
    Tại sao: Attention cần đủ channel diversity để chọn lọc
    """
    def __init__(self, params):
        super().__init__()
        self.ft_chns = params['feature_chns']
        self.dropout = params['dropout']
        
        self.in_conv = ResConvBlock(params['in_chns'], self.ft_chns[0], 
                                     self.dropout[0], use_se=True)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        )
    
    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderV3(nn.Module):
    """
    Decoder với Attention Gates + ASPP + Deep Supervision
    
    So với V1:
    - Skip: Direct concat → AttentionGate filtered → giảm noise
    - Bottleneck: Plain → ASPP → multi-scale context
    - DS weight: Giảm dần theo training (thay vì cố định 0.4)
    """
    def __init__(self, params):
        super().__init__()
        ft = params['feature_chns']
        self.n_class = params['class_num']
        
        # ASPP tại bottleneck
        self.aspp = ASPP(ft[4], ft[4])
        
        # Attention Gates cho mỗi skip connection
        self.ag3 = AttentionGate(ft[4], ft[3], ft[3] // 2)
        self.ag2 = AttentionGate(ft[3], ft[2], ft[2] // 2)
        self.ag1 = AttentionGate(ft[2], ft[1], ft[1] // 2)
        self.ag0 = AttentionGate(ft[1], ft[0], ft[0] // 2)
        
        # Upsampling + Conv
        self.up4 = nn.ConvTranspose2d(ft[4], ft[3], kernel_size=2, stride=2)
        self.conv4 = ResConvBlock(ft[3] * 2, ft[3], 0.0, use_se=True)
        
        self.up3 = nn.ConvTranspose2d(ft[3], ft[2], kernel_size=2, stride=2)
        self.conv3 = ResConvBlock(ft[2] * 2, ft[2], 0.0, use_se=True)
        
        self.up2 = nn.ConvTranspose2d(ft[2], ft[1], kernel_size=2, stride=2)
        self.conv2 = ResConvBlock(ft[1] * 2, ft[1], 0.0, use_se=True)
        
        self.up1 = nn.ConvTranspose2d(ft[1], ft[0], kernel_size=2, stride=2)
        self.conv1 = ResConvBlock(ft[0] * 2, ft[0], 0.0, use_se=True)
        
        self.out_conv = nn.Conv2d(ft[0], self.n_class, kernel_size=1)
        
        # Deep supervision heads
        self.ds_out3 = nn.Conv2d(ft[3], self.n_class, kernel_size=1)
        self.ds_out2 = nn.Conv2d(ft[2], self.n_class, kernel_size=1)
        self.ds_out1 = nn.Conv2d(ft[1], self.n_class, kernel_size=1)
    
    def forward(self, features, return_deep_supervision=False):
        x0, x1, x2, x3, x4 = features
        target_shape = x0.shape[2:]
        
        x4 = self.aspp(x4)
        
        # Decoder level 4→3: AG filter x3 trước khi concat
        x3_att = self.ag3(g=x4, x=x3)
        d3 = torch.cat([self.up4(x4), x3_att], dim=1)
        d3 = self.conv4(d3)
        if return_deep_supervision:
            ds3 = F.interpolate(self.ds_out3(d3), size=target_shape,
                               mode='bilinear', align_corners=True)
        
        x2_att = self.ag2(g=d3, x=x2)
        d2 = torch.cat([self.up3(d3), x2_att], dim=1)
        d2 = self.conv3(d2)
        if return_deep_supervision:
            ds2 = F.interpolate(self.ds_out2(d2), size=target_shape,
                               mode='bilinear', align_corners=True)
        
        x1_att = self.ag1(g=d2, x=x1)
        d1 = torch.cat([self.up2(d2), x1_att], dim=1)
        d1 = self.conv2(d1)
        if return_deep_supervision:
            ds1 = F.interpolate(self.ds_out1(d1), size=target_shape,
                               mode='bilinear', align_corners=True)
        
        x0_att = self.ag0(g=d1, x=x0)
        d0 = torch.cat([self.up1(d1), x0_att], dim=1)
        d0 = self.conv1(d0)
        
        output = self.out_conv(d0)
        
        if return_deep_supervision:
            return output, ds1, ds2, ds3
        return output


class ImprovedUNetV3(nn.Module):
    """
    Tổng hợp tất cả cải tiến:
    ┌─────────────────┬──────────────────┬───────────────────┐
    │   Component     │     V1 (cũ)      │     V3 (mới)      │
    ├─────────────────┼──────────────────┼───────────────────┤
    │ Channels        │ [16,32,64,128,256]│ [32,64,128,256,512]│
    │ Encoder Att.    │ CBAM (level 3,4) │ SE (all levels)   │
    │ Skip Connection │ Direct concat    │ Attention Gate     │
    │ Bottleneck      │ Plain conv       │ ASPP              │
    │ Normalization   │ BatchNorm        │ GroupNorm         │
    │ Dropout         │ [.05,.1,.2,.3,.5]│ [.05,.1,.15,.2,.3]│
    │ Deep Supervision│ Weight cố định   │ Weight giảm dần   │
    │ Params ước tính │ ~3M              │ ~15M              │
    └─────────────────┴──────────────────┴───────────────────┘
    """
    def __init__(self, in_chns, class_num, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        params = {
            'in_chns': in_chns,
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.05, 0.1, 0.15, 0.2, 0.3],  # Giảm để tránh over-regularization
            'class_num': class_num,
        }
        
        self.encoder = EncoderV3(params)
        self.decoder = DecoderV3(params)
        self._init_weights()
    
    def _init_weights(self):
        """He initialization cho LeakyReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.encoder(x)
        if self.training and self.deep_supervision:
            return self.decoder(features, return_deep_supervision=True)
        return self.decoder(features, return_deep_supervision=False)


# =====================================================================
#                    TRAINING FUNCTIONS
# =====================================================================

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, 
                    "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    CẢI TIẾN: EMA decay scheduling
    
    V1: alpha cố định = 0.99 suốt training
    V3: alpha tăng từ 0.99 → 0.999 theo thời gian
    
    Lý do: 
    - Đầu training: teacher cần update nhanh (alpha thấp) để theo kịp student
    - Cuối training: teacher cần ổn định (alpha cao) cho pseudo-labels tốt hơn
    """
    max_alpha = min(alpha + 0.009, 0.999)
    scheduled_alpha = alpha + (max_alpha - alpha) * min(global_step / 10000.0, 1.0)
    scheduled_alpha = min(1 - 1 / (global_step + 1), scheduled_alpha)
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(scheduled_alpha).add_(param.data, alpha=(1 - scheduled_alpha))


def get_cosine_lr(base_lr, iter_num, max_iterations, warmup_iters=1000):
    """
    CẢI TIẾN: Cosine Annealing LR + Warmup
    
    V1: lr = base_lr * (1 - iter/max_iter)^0.9 (polynomial, monotonic)
    V3: warmup linear 0→base_lr, sau đó cosine decay
    
    Tại sao tốt hơn:
    - Warmup: tránh gradient lớn ở đầu khi attention modules random
    - Cosine: smooth hơn polynomial, proven tốt hơn trong literature
    """
    if iter_num < warmup_iters:
        return base_lr * iter_num / warmup_iters
    progress = (iter_num - warmup_iters) / (max_iterations - warmup_iters)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def create_model(ema=False):
    model = ImprovedUNetV3(in_chns=1, class_num=args.num_classes, deep_supervision=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = create_model(ema=False).cuda()
    ema_model = create_model(ema=True).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total params: {total_params:,}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
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

    # CẢI TIẾN: AdamW thay SGD
    # AdamW có adaptive learning rate → hội tụ nhanh hơn SGD
    # weight_decay=0.01 cho regularization (thay L2 trong SGD)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr * 0.1, 
                            weight_decay=0.01, betas=(0.9, 0.999))

    # Loss V3
    combined_loss = CombinedLossV3(
        n_classes=num_classes,
        el_gamma=args.el_loss_gamma,
        hd_weight=args.hd_loss_weight
    )
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"{len(trainloader)} iterations per epoch")

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            # Student forward
            model.train()
            model_output = model(volume_batch)
            
            if isinstance(model_output, tuple):
                outputs, ds1, ds2, ds3 = model_output
            else:
                outputs = model_output
                ds1 = ds2 = ds3 = None
            
            outputs_soft = torch.softmax(outputs, dim=1)

            # Teacher forward (with noise)
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                if isinstance(ema_output, tuple):
                    ema_output = ema_output[0]

            # MC Dropout uncertainty estimation
            T = 8
            _, _, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
            
            ema_model.train()
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

            # === Supervised Loss V3 ===
            supervised_loss, loss_ce, loss_region = combined_loss(
                outputs[:args.labeled_bs],
                outputs_soft[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                iter_num, max_iterations
            )
            
            # Deep supervision với weight GIẢM DẦN (V1 cố định 0.4)
            if ds1 is not None:
                ds_loss = 0.0
                for ds_out in [ds1, ds2, ds3]:
                    ds_soft = torch.softmax(ds_out[:args.labeled_bs], dim=1)
                    ds_loss += dice_loss(ds_soft, label_batch[:args.labeled_bs].unsqueeze(1))
                ds_loss /= 3.0
                # Giảm từ 0.4 → 0.05 theo training progress
                ds_weight = 0.4 * (1.0 - min(iter_num / max_iterations, 0.8))
                supervised_loss = supervised_loss + ds_weight * ds_loss

            # === Mixed Consistency (MSE + KL) ===
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            # MSE component
            consistency_dist_mse = losses.softmax_mse_loss(
                outputs[args.labeled_bs:], ema_output
            )
            # KL component (tốt hơn MSE cho probability distributions)
            student_log_soft = F.log_softmax(outputs[args.labeled_bs:], dim=1)
            teacher_soft = F.softmax(ema_output, dim=1)
            consistency_dist_kl = F.kl_div(student_log_soft, teacher_soft, reduction='none')
            
            kl_w = args.kl_consistency_weight
            consistency_dist = (1 - kl_w) * consistency_dist_mse + kl_w * consistency_dist_kl
            
            # Uncertainty masking
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            # Total loss
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # Cosine LR + Warmup
            lr_ = get_cosine_lr(base_lr * 0.1, iter_num, max_iterations, args.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # Logging
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

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                pred_vis = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # Validation
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
                    writer.add_scalar(f'info/val_{class_i+1}_dice',
                                     metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f'info/val_{class_i+1}_hd95',
                                     metric_list[class_i, 1], iter_num)

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

                logging.info(
                    f'iteration {iter_num}: mean_dice={performance:.4f}, '
                    f'mean_hd95={mean_hd95:.4f}, best={best_performance:.4f}'
                )
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_mode_path)

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
        filename=snapshot_path + "/log.txt", level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)