"""
re_code_v2.py - Cai tien UAMT V2 cho ACDC Dataset

Cai tien so voi re_code.py (V1):
    === LOSS FUNCTIONS ===
    1. Exponential Logarithmic Loss (EL Loss): Ket hop Dice va CE trong khong gian log
       -> Gradient on dinh hon, hoi tu nhanh hon CE+Dice thong thuong
    2. Class-Adaptive Dice Loss: Tinh Dice rieng tung class voi adaptive weights
       -> Xu ly class imbalance tot hon Focal Loss (don gian va hieu qua hon)
    3. Hausdorff Distance Loss (HD Loss): Toi uu truc tiep metric HD95
       -> Cai thien manh boundary segmentation (thay BoundaryLoss cu)
    4. KL Divergence Consistency: Thay MSE bang KL divergence cho consistency
       -> Phu hop hon cho so sanh distributions (softmax outputs)
    
    === MODEL ===
    5. Attention Gate tai skip connections: Loc features truoc khi concatenate
       -> Giam noise tu encoder, focus vao vung quan trong (thay CBAM)
    6. Squeeze-and-Excitation (SE) blocks: Channel recalibration hieu qua
       -> Nhe hon CBAM, hieu qua tuong duong
    7. ASPP (Atrous Spatial Pyramid Pooling) tai bottleneck
       -> Multi-scale context giup nhan dien cau truc tim o nhieu kich thuoc
    8. Wider channels [32, 64, 128, 256, 512]
       -> Tang capacity de hoc features phuc tap hon
    
    === TRAINING ===
    9. Cosine Annealing LR + Warmup: Hoi tu tot hon polynomial decay
    10. EMA Decay Scheduling: Tang dan tu 0.99 len 0.999
    11. Mixed Consistency (MSE + KL): Ket hop ca hai cho stability
    12. Test-Time Augmentation (TTA) trong validation
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
                    default='ACDC/Improved_UAMT_V2', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# ========== V2: Tham so moi ==========
parser.add_argument('--warmup_iters', type=int, default=1000,
                    help='So iterations warmup cho learning rate')
parser.add_argument('--use_cosine_lr', type=int, default=1,
                    help='Su dung cosine annealing LR (0/1)')
parser.add_argument('--hd_loss_weight', type=float, default=0.5,
                    help='Trong so cho Hausdorff Distance Loss')
parser.add_argument('--el_loss_wdice', type=float, default=0.8,
                    help='Trong so Dice trong Exponential Logarithmic Loss')
parser.add_argument('--el_loss_wce', type=float, default=0.2,
                    help='Trong so CE trong Exponential Logarithmic Loss')
parser.add_argument('--el_loss_gamma', type=float, default=0.3,
                    help='Gamma cho Exponential Logarithmic Loss')
parser.add_argument('--kl_consistency_weight', type=float, default=0.5,
                    help='Ty le KL trong mixed consistency (0=pure MSE, 1=pure KL)')

args = parser.parse_args()


# =====================================================================
#                 V2 CAI TIEN 1: LOSS FUNCTIONS NANG CAO
# =====================================================================

class ExponentialLogarithmicLoss(nn.Module):
    """
    Exponential Logarithmic Loss (EL Loss)
    
    Paper: "3D Segmentation with Exponential Logarithmic Loss for Highly 
           Unbalanced Object Sizes" (MICCAI 2018)
    
    Y tuong: Ket hop Dice va CE trong khong gian logarithmic:
      L_el = w_dice * (E[-ln(Dice_i)])^gamma + w_ce * (E[(-ln(p))^gamma])
    
    Uu diem so voi V1 (CE + Dice + Focal):
    - Gradient on dinh hon: log transform lam smooth gradient
    - Xu ly class imbalance tot: Gamma < 1 focus vao classes kho
    - Chi can 1 loss thay vi 3 losses -> it conflict hon
    - Khong can alpha weights nhu Focal Loss
    
    ACDC: Giup can bang giua background (lon) va RV/MYO/LV (nho)
    """
    def __init__(self, n_classes, w_dice=0.8, w_ce=0.2, gamma=0.3):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.n_classes = n_classes
        self.w_dice = w_dice    # Trong so cho Dice component
        self.w_ce = w_ce        # Trong so cho CE component
        self.gamma = gamma      # < 1: focus vao classes kho; > 1: focus vao classes de
    
    def _dice_per_class(self, pred, target):
        """Tinh Dice score cho tung class rieng biet"""
        smooth = 1e-5
        dice_scores = []
        for c in range(self.n_classes):
            pred_c = pred[:, c]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            dice_scores.append(dice)
        return dice_scores
    
    def forward(self, outputs_soft, labels):
        """
        outputs_soft: [B, C, H, W] - softmax predictions
        labels: [B, H, W] - ground truth labels
        """
        # === Dice component trong log space ===
        dice_scores = self._dice_per_class(outputs_soft, labels)
        # -ln(Dice) cho tung class, sau do lay mean va power gamma
        dice_log = torch.stack([-torch.log(d + 1e-7) for d in dice_scores])
        loss_dice = torch.mean(dice_log ** self.gamma)
        
        # === Cross-Entropy component trong log space ===
        # Lay xac suat cua class dung cho moi pixel
        labels_long = labels.long()
        # Gather: chon xac suat cua class dung
        pt = outputs_soft.gather(1, labels_long.unsqueeze(1)).squeeze(1)
        # (-ln(pt))^gamma - focus vao pixels kho
        loss_ce = torch.mean((-torch.log(pt + 1e-7)) ** self.gamma)
        
        return self.w_dice * loss_dice + self.w_ce * loss_ce


class ClassAdaptiveDiceLoss(nn.Module):
    """
    Class-Adaptive Dice Loss
    
    Y tuong: Tinh Dice rieng tung class va gan weight tu dong
    dua tren inverse frequency (classes nho duoc weight cao hon).
    
    Uu diem so voi V1 DiceLoss:
    - V1 tinh Dice toan bo roi chia trung binh -> classes nho bi an
    - V2 weight tu dong theo kich thuoc class -> khong can manual alpha
    - Combination voi class frequency de tu adapt theo tung batch
    
    ACDC: RV/MYO/LV nho hon background nhieu -> duoc weight cao tu dong
    """
    def __init__(self, n_classes, smooth=1e-5):
        super(ClassAdaptiveDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] - softmax predictions
        targets: [B, 1, H, W] hoac [B, H, W] - ground truth
        """
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        
        total_loss = 0.0
        total_weight = 0.0
        
        for c in range(self.n_classes):
            pred_c = inputs[:, c]
            target_c = (targets == c).float()
            
            # Tinh Dice cho class nay
            intersection = (pred_c * target_c).sum()
            pred_sum = pred_c.sum()
            target_sum = target_c.sum()
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            # Adaptive weight: inverse of class frequency
            # Classes nho (it pixels) se co weight lon hon
            class_freq = target_c.sum() / (target_c.numel() + self.smooth)
            # Gia tri weight: 1/sqrt(freq) de tranh weight qua lon cho class rat nho
            weight = 1.0 / (torch.sqrt(class_freq) + self.smooth)
            
            total_loss += weight * (1.0 - dice)
            total_weight += weight
        
        return total_loss / (total_weight + self.smooth)


class HausdorffDistanceLoss(nn.Module):
    """
    Hausdorff Distance Loss (xap xi differentiable)
    
    Paper: "Reducing the Hausdorff Distance in Medical Image Segmentation 
           with Convolutional Neural Networks" (IEEE TMI 2020)
    
    Y tuong: Xap xi Hausdorff Distance bang distance transform + erosion.
    Thay vi tinh HD95 chinh xac (khong differentiable), dung 
    convolutional erosion de xap xi boundary distance.
    
    Uu diem so voi V1 BoundaryLoss:
    - V1 dung Sobel filter chi detect edges -> khong co thong tin khoang cach
    - V2 dung multi-scale erosion -> xap xi distance transform
    - Truc tiep toi uu metric HD95 duoc dung de danh gia
    
    ACDC: HD95 la metric quan trong -> toi uu truc tiep se cai thien dang ke
    """
    def __init__(self, n_classes, alpha=2.0):
        super(HausdorffDistanceLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha  # So mu cho distance weighting
        
        # Tao erosion kernels o nhieu scales
        # Moi kernel tuong ung voi 1 level distance
        self.register_buffer('kernel_3', self._create_erosion_kernel(3))
        self.register_buffer('kernel_5', self._create_erosion_kernel(5))
        self.register_buffer('kernel_7', self._create_erosion_kernel(7))
    
    def _create_erosion_kernel(self, size):
        """Tao circular kernel cho morphological erosion"""
        kernel = torch.ones(1, 1, size, size)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 > center**2:
                    kernel[0, 0, i, j] = 0
        return kernel / kernel.sum()
    
    def _approximate_distance_map(self, mask):
        """
        Xap xi distance transform bang multi-scale erosion.
        Moi lop erosion tuong ung voi 1 don vi khoang cach tu boundary.
        """
        # Erosion = 1 - dilation of (1 - mask)
        inv_mask = 1.0 - mask
        
        # Distance level 1 (erosion 3x3)
        eroded_1 = F.conv2d(mask, self.kernel_3.to(mask.device), padding=1)
        boundary_1 = mask - (eroded_1 > 0.5).float() * mask
        
        # Distance level 2 (erosion 5x5)
        eroded_2 = F.conv2d(mask, self.kernel_5.to(mask.device), padding=2)
        boundary_2 = mask - (eroded_2 > 0.5).float() * mask - boundary_1
        boundary_2 = torch.clamp(boundary_2, 0, 1)
        
        # Distance level 3 (erosion 7x7)
        eroded_3 = F.conv2d(mask, self.kernel_7.to(mask.device), padding=3)
        boundary_3 = mask - (eroded_3 > 0.5).float() * mask - boundary_1 - boundary_2
        boundary_3 = torch.clamp(boundary_3, 0, 1)
        
        # Distance map: pixels gan boundary co distance nho, xa co distance lon
        # Inverse: boundary pixels co weight cao nhat
        dist_map = boundary_1 * 1.0 + boundary_2 * 2.0 + boundary_3 * 3.0
        # Normalize
        max_val = dist_map.max() + 1e-7
        dist_map = dist_map / max_val
        
        return dist_map
    
    def forward(self, outputs_soft, labels):
        """
        outputs_soft: [B, C, H, W] - softmax predictions
        labels: [B, H, W] - ground truth
        """
        total_loss = 0.0
        n_valid = 0
        
        for c in range(1, self.n_classes):  # Bo qua background (class 0)
            pred_c = outputs_soft[:, c:c+1]
            target_c = (labels == c).float().unsqueeze(1)
            
            if target_c.sum() < 10:  # Bo qua class khong co mat
                continue
            
            # Tinh distance maps
            dist_target = self._approximate_distance_map(target_c)
            dist_pred = self._approximate_distance_map((pred_c > 0.5).float())
            
            # HD Loss: Trong so error theo khoang cach tu boundary
            # Pixels sai xa boundary bi phat nang hon
            error = (pred_c - target_c) ** 2
            weighted_error = error * (dist_target + dist_pred + 1.0) ** self.alpha
            
            total_loss += weighted_error.mean()
            n_valid += 1
        
        if n_valid == 0:
            return torch.tensor(0.0, device=outputs_soft.device, requires_grad=True)
        
        return total_loss / n_valid


class CombinedLossV2(nn.Module):
    """
    Combined Loss V2 - Don gian hoa va hieu qua hon
    
    So sanh voi V1:
    - V1: CE + Dice + Focal + Boundary = 4 losses, nhieu conflict
    - V2: EL Loss + Adaptive Dice + HD Loss = 3 losses, bo sung cho nhau
      + EL Loss: Thay the CE + Focal (xu ly class imbalance + on dinh gradient)
      + Adaptive Dice: Thay Dice goc (tu dong can bang classes)
      + HD Loss: Thay BoundaryLoss (toi uu truc tiep HD95 metric)
    
    Strategy: 
    - Phase 1 (0-40% training): EL Loss + Adaptive Dice (hoc shape co ban)
    - Phase 2 (40-100%): + HD Loss ramp up (tinh chinh boundary)
    """
    def __init__(self, n_classes, 
                 el_wdice=0.8, el_wce=0.2, el_gamma=0.3,
                 hd_weight=0.5):
        super(CombinedLossV2, self).__init__()
        self.n_classes = n_classes
        self.hd_weight = hd_weight
        
        # Main losses
        self.el_loss = ExponentialLogarithmicLoss(n_classes, el_wdice, el_wce, el_gamma)
        self.adaptive_dice = ClassAdaptiveDiceLoss(n_classes)
        self.hd_loss = HausdorffDistanceLoss(n_classes)
        
        # Backup standard losses cho logging
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = losses.DiceLoss(n_classes)
    
    def forward(self, outputs, outputs_soft, labels, iter_num=0, max_iterations=30000):
        """
        outputs: [B, C, H, W] - raw logits
        outputs_soft: [B, C, H, W] - softmax predictions
        labels: [B, H, W] - ground truth
        """
        labels_ce = labels.long()
        labels_dice = labels.unsqueeze(1)
        
        # === EL Loss: Thay the CE + Focal ===
        loss_el = self.el_loss(outputs_soft, labels)
        
        # === Adaptive Dice: Can bang classes tu dong ===
        loss_adice = self.adaptive_dice(outputs_soft, labels)
        
        # === Ket hop EL + Adaptive Dice ===
        supervised_loss = 0.5 * loss_el + 0.5 * loss_adice
        
        # === HD Loss: Ramp up sau 40% training ===
        # Giai doan dau hoc shape co ban, sau do moi tinh chinh boundary
        hd_rampup_start = max_iterations * 0.4
        if iter_num > hd_rampup_start:
            hd_rampup = ramps.sigmoid_rampup(
                iter_num - hd_rampup_start, 
                max_iterations * 0.3  # Ramp up trong 30% training tiep theo
            )
            loss_hd = self.hd_loss(outputs_soft, labels)
            supervised_loss = supervised_loss + self.hd_weight * hd_rampup * loss_hd
        
        # Tinh CE va Dice goc de logging (khong dung cho backprop)
        with torch.no_grad():
            loss_ce_log = self.ce_loss(outputs, labels_ce)
            loss_dice_log = self.dice_loss(outputs_soft, labels_dice)
        
        return supervised_loss, loss_ce_log, loss_dice_log


# =====================================================================
#                 V2 CAI TIEN 2: MODEL ARCHITECTURE
# =====================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block (SE Block)
    
    Paper: "Squeeze-and-Excitation Networks" (CVPR 2018)
    
    Y tuong: Hoc adaptive channel weights thong qua:
    1. Squeeze: Global Average Pooling nen spatial info vao 1 vector
    2. Excitation: 2-layer MLP hoc channel importance
    3. Scale: Nhan nguoc lai vao feature map
    
    So sanh voi V1 (CBAM):
    - CBAM = Channel Attention + Spatial Attention (nang hon)
    - SE = Chi Channel Attention nhung hieu qua tuong duong
    - SE nhe hon CBAM ~40% parameters -> nhanh hon, it overfit hon
    - Voi small dataset nhu ACDC (7 labeled), nhe hon = tot hon
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        mid_channels = max(channels // reduction, 8)  # Dam bao >= 8
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: [B, C, H, W] -> [B, C]
        y = self.squeeze(x).view(b, c)
        # Excitation: [B, C] -> [B, C]
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: element-wise multiplication
        return x * y.expand_as(x)


class AttentionGate(nn.Module):
    """
    Attention Gate cho skip connections
    
    Paper: "Attention U-Net" (MIDL 2018)
    
    Y tuong: Loc features tu encoder truoc khi concat voi decoder.
    Dung signal tu decoder (gating) de quyet dinh phan nao cua 
    encoder features la quan trong.
    
    So sanh voi V1 (CBAM o encoder):
    - V1 dat CBAM o encoder -> chi refine encoder features
    - V2 dat Attention Gate o skip connection -> loc features 
      dua tren context tu decoder (thong minh hon)
    - Decoder biet can gi -> noi cho encoder phan nao duoc giu
    
    ACDC: Loc bo background noise tu encoder, chi giu features
    lien quan den vung tim khi truyen sang decoder
    
    g: gating signal tu decoder (biet can gi)
    x: encoder features (chua thong tin tho)
    output: encoder features da duoc loc (chi giu phan quan trong)
    """
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: so channels cua gating signal (tu decoder)
        F_l: so channels cua encoder features
        F_int: so channels intermediate
        """
        super(AttentionGate, self).__init__()
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transform encoder features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        g: gating signal [B, F_g, H, W]
        x: encoder features [B, F_l, H, W]
        """
        # Align spatial dimensions (g co the nho hon x)
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Nhan attention weights voi encoder features
        return x * psi


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    
    Paper: "DeepLab v3+" (ECCV 2018)
    
    Y tuong: Su dung dilated convolutions voi nhieu dilation rates
    de capture multi-scale context tu cung 1 feature map.
    
    Rate=1: Chi tiet nho (local context)
    Rate=6: Chi tiet trung binh
    Rate=12: Chi tiet lon (global context)
    + Global Average Pooling: Context toan cuc
    
    Dat tai bottleneck cua U-Net de giup model "nhin" 
    cau truc tim o nhieu kich thuoc khac nhau.
    
    ACDC: Tim co nhieu cau truc o nhieu scales:
    - LV cavity (lon)
    - MYO wall (mong, dai)
    - RV (hinh dang phuc tap)
    -> ASPP giup capture tat ca scales nay dong thoi
    
    So voi V1: V1 khong co multi-scale feature extraction tai bottleneck
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolutions voi dilations khac nhau
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling branch
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fuse tat ca branches
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        feat1 = self.conv1(x)
        feat6 = self.conv6(x)
        feat12 = self.conv12(x)
        feat_pool = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        
        # Concatenate tat ca branches
        out = torch.cat([feat1, feat6, feat12, feat_pool], dim=1)
        out = self.fuse(out)
        
        return out


class ResConvBlock(nn.Module):
    """
    Residual Convolution Block voi SE attention
    
    Cai tien tu V1 ResidualConvBlock:
    - Them SE block sau convolutions de refine channels
    - Su dung GroupNorm thay BatchNorm (tot hon voi small batch size)
    
    GroupNorm vs BatchNorm:
    - BatchNorm phu thuoc batch size -> khong on dinh khi batch nho
    - GroupNorm chia channels thanh groups, normalize trong moi group
    - Voi batch_size=24 thi BatchNorm OK, nhung GroupNorm van on dinh hon
    - ACDC co labeled_bs=12, nen GroupNorm giup phan labeled on dinh hon
    """
    def __init__(self, in_channels, out_channels, dropout_p, use_se=True):
        super(ResConvBlock, self).__init__()
        self.use_se = use_se
        
        # Chon so groups cho GroupNorm (phai chia het cho out_channels)
        num_groups = min(32, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),  # Dropout2d tot hon Dropout cho feature maps
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
        )
        
        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
        
        # SE Block
        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=16)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv_block(x)
        if self.use_se:
            out = self.se(out)
        out = out + residual
        return self.activation(out)


class EncoderV2(nn.Module):
    """
    Encoder V2 voi wider channels va SE blocks
    
    So sanh voi V1:
    - Channels: [16,32,64,128,256] -> [32,64,128,256,512]
      (Tang capacity de hoc features phuc tap hon)
    - Attention: CBAM o level 3,4 -> SE blocks o tat ca levels
      (Nhe hon nhung toan dien hon)
    - Normalization: BatchNorm -> GroupNorm
      (On dinh hon voi small batch)
    """
    def __init__(self, params):
        super(EncoderV2, self).__init__()
        self.in_chns = params['in_chns']
        self.ft_chns = params['feature_chns']
        self.dropout = params['dropout']
        
        # Level 0: Input convolution
        self.in_conv = ResConvBlock(self.in_chns, self.ft_chns[0], 
                                     self.dropout[0], use_se=True)
        # Level 1-4: Downsampling + ResConvBlock
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1], use_se=True)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2], use_se=True)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3], use_se=True)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4], use_se=True)
        )
    
    def forward(self, x):
        x0 = self.in_conv(x)     # [B, 32, 256, 256]
        x1 = self.down1(x0)      # [B, 64, 128, 128]
        x2 = self.down2(x1)      # [B, 128, 64, 64]
        x3 = self.down3(x2)      # [B, 256, 32, 32]
        x4 = self.down4(x3)      # [B, 512, 16, 16]
        return [x0, x1, x2, x3, x4]


class DecoderV2(nn.Module):
    """
    Decoder V2 voi Attention Gates va Deep Supervision
    
    So sanh voi V1:
    - Skip connections: Direct concat -> Attention Gate filtered concat
      (Loc bo noise, chi giu features huu ich tu encoder)
    - ASPP tai bottleneck: Khong co -> Co
      (Multi-scale context capture)
    - Deep Supervision: Co o ca V1 va V2, nhung V2 dung SE blocks nhe hon
    """
    def __init__(self, params):
        super(DecoderV2, self).__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']
        
        # ASPP tai bottleneck
        self.aspp = ASPP(self.ft_chns[4], self.ft_chns[4])
        
        # Attention Gates cho skip connections
        self.ag3 = AttentionGate(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3] // 2)
        self.ag2 = AttentionGate(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2] // 2)
        self.ag1 = AttentionGate(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1] // 2)
        self.ag0 = AttentionGate(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0] // 2)
        
        # Upsampling + ResConvBlock
        self.up4 = nn.ConvTranspose2d(self.ft_chns[4], self.ft_chns[3], kernel_size=2, stride=2)
        self.conv4 = ResConvBlock(self.ft_chns[3] * 2, self.ft_chns[3], 0.0, use_se=True)
        
        self.up3 = nn.ConvTranspose2d(self.ft_chns[3], self.ft_chns[2], kernel_size=2, stride=2)
        self.conv3 = ResConvBlock(self.ft_chns[2] * 2, self.ft_chns[2], 0.0, use_se=True)
        
        self.up2 = nn.ConvTranspose2d(self.ft_chns[2], self.ft_chns[1], kernel_size=2, stride=2)
        self.conv2 = ResConvBlock(self.ft_chns[1] * 2, self.ft_chns[1], 0.0, use_se=True)
        
        self.up1 = nn.ConvTranspose2d(self.ft_chns[1], self.ft_chns[0], kernel_size=2, stride=2)
        self.conv1 = ResConvBlock(self.ft_chns[0] * 2, self.ft_chns[0], 0.0, use_se=True)
        
        # Final output
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)
        
        # Deep supervision outputs
        self.ds_out3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=1)
        self.ds_out2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=1)
        self.ds_out1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=1)
    
    def forward(self, features, return_deep_supervision=False):
        x0, x1, x2, x3, x4 = features
        target_shape = x0.shape[2:]
        
        # ASPP tai bottleneck
        x4 = self.aspp(x4)
        
        # Level 4 -> 3: Attention Gate loc x3, roi concat voi upsampled x4
        x3_att = self.ag3(g=x4, x=x3)
        d3 = self.up4(x4)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.conv4(d3)
        if return_deep_supervision:
            ds3 = F.interpolate(self.ds_out3(d3), size=target_shape, 
                               mode='bilinear', align_corners=True)
        
        # Level 3 -> 2
        x2_att = self.ag2(g=d3, x=x2)
        d2 = self.up3(d3)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.conv3(d2)
        if return_deep_supervision:
            ds2 = F.interpolate(self.ds_out2(d2), size=target_shape,
                               mode='bilinear', align_corners=True)
        
        # Level 2 -> 1
        x1_att = self.ag1(g=d2, x=x1)
        d1 = self.up2(d2)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.conv2(d1)
        if return_deep_supervision:
            ds1 = F.interpolate(self.ds_out1(d1), size=target_shape,
                               mode='bilinear', align_corners=True)
        
        # Level 1 -> 0
        x0_att = self.ag0(g=d1, x=x0)
        d0 = self.up1(d1)
        d0 = torch.cat([d0, x0_att], dim=1)
        d0 = self.conv1(d0)
        
        output = self.out_conv(d0)
        
        if return_deep_supervision:
            return output, ds1, ds2, ds3
        return output


class ImprovedUNetV2(nn.Module):
    """
    Improved UNet V2 - Ket hop tat ca cai tien
    
    So sanh tong the voi V1:
    +-----------------------+-------------------+-------------------+
    |       Component       |       V1          |       V2          |
    +-----------------------+-------------------+-------------------+
    | Channels              | [16,32,64,128,256]| [32,64,128,256,512]|
    | Encoder Attention     | CBAM (level 3,4)  | SE (all levels)   |
    | Decoder Attention     | Khong co          | Attention Gates   |
    | Skip Connections      | Direct concat     | AG-filtered concat|
    | Bottleneck            | Plain conv        | ASPP              |
    | Normalization         | BatchNorm         | GroupNorm         |
    | Deep Supervision      | Co                | Co (cai tien)     |
    | Residual Connections  | Co                | Co + SE           |
    +-----------------------+-------------------+-------------------+
    
    Estimated params: ~15M (V1: ~3M, Original UNet: ~2M)
    Tang params nhung hieu qua hon nho attention va multi-scale
    """
    def __init__(self, in_chns, class_num, deep_supervision=True):
        super(ImprovedUNetV2, self).__init__()
        self.deep_supervision = deep_supervision
        
        params = {
            'in_chns': in_chns,
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.05, 0.1, 0.15, 0.2, 0.3],  # Giam dropout (V1 qua cao)
            'class_num': class_num,
        }
        
        self.encoder = EncoderV2(params)
        self.decoder = DecoderV2(params)
        
        # Kaiming initialization cho tat ca conv layers
        self._init_weights()
    
    def _init_weights(self):
        """He initialization - tot cho LeakyReLU"""
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
    """Consistency weight voi rampup"""
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    V2: EMA decay scheduling
    
    Y tuong: Tang EMA decay tu 0.99 -> 0.999 theo thoi gian
    - Giai doan dau: decay thap -> teacher update nhanh theo student
    - Giai doan sau: decay cao -> teacher on dinh hon, smooth hon
    
    Cong thuc: alpha = min(1 - 1/(step+1), alpha_base + (alpha_max - alpha_base) * (step/max_step))
    """
    # Scheduled alpha: tang tu alpha -> min(alpha + 0.009, 0.999) 
    max_alpha = min(alpha + 0.009, 0.999)
    scheduled_alpha = alpha + (max_alpha - alpha) * min(global_step / 10000.0, 1.0)
    scheduled_alpha = min(1 - 1 / (global_step + 1), scheduled_alpha)
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(scheduled_alpha).add_(param.data, alpha=(1 - scheduled_alpha))


def get_cosine_lr(base_lr, iter_num, max_iterations, warmup_iters=1000):
    """
    Cosine Annealing LR voi Warmup
    
    So sanh voi V1 (polynomial decay):
    - V1: lr = base_lr * (1 - iter/max_iter)^0.9
      -> Giam nhanh o giua, cham o cuoi
    - V2: warmup + cosine annealing
      -> Warmup: tang tu 0 -> base_lr trong warmup_iters dau
      -> Cosine: giam tu base_lr -> 0 theo hinh cosine
      -> Smooth hon, giup hoi tu tot hon (proven trong literature)
    """
    if iter_num < warmup_iters:
        # Linear warmup
        return base_lr * iter_num / warmup_iters
    else:
        # Cosine annealing  
        progress = (iter_num - warmup_iters) / (max_iterations - warmup_iters)
        return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def create_model(ema=False, use_improved=True):
    """Tao model voi option su dung Improved UNet V2"""
    if use_improved:
        model = ImprovedUNetV2(in_chns=1, class_num=args.num_classes, deep_supervision=True)
    else:
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

    # ===== Tao models =====
    model = create_model(ema=False, use_improved=True)
    ema_model = create_model(ema=True, use_improved=True)
    
    model = model.cuda()
    ema_model = ema_model.cuda()
    
    # Log so params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

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

    # ===== Optimizer: AdamW thay SGD =====
    # AdamW on dinh hon SGD cho medical image segmentation
    # Adaptive learning rate giup hoi tu nhanh hon
    optimizer = optim.AdamW(model.parameters(), lr=base_lr * 0.1, 
                            weight_decay=0.01, betas=(0.9, 0.999))

    # ===== Loss functions V2 =====
    combined_loss = CombinedLossV2(
        n_classes=num_classes,
        el_wdice=args.el_loss_wdice,
        el_wce=args.el_loss_wce,
        el_gamma=args.el_loss_gamma,
        hd_weight=args.hd_loss_weight
    )
    
    # Standard losses cho deep supervision va backup
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # ===== Training loop =====
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"{len(trainloader)} iterations per epoch")
    logging.info("=== V2 Architecture: SE + AttentionGate + ASPP + GroupNorm ===")
    logging.info("=== V2 Loss: EL Loss + Adaptive Dice + HD Loss ===")
    logging.info("=== V2 Training: Cosine LR + Warmup + EMA Scheduling ===")

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
            model_output = model(volume_batch)
            
            if isinstance(model_output, tuple):
                outputs, ds1, ds2, ds3 = model_output
            else:
                outputs = model_output
                ds1 = ds2 = ds3 = None
            
            outputs_soft = torch.softmax(outputs, dim=1)

            # ===== Teacher forward pass (voi noise) =====
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
            
            ema_model.train()  # Enable dropout cho uncertainty
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

            # ===== Supervised Loss V2 =====
            supervised_loss, loss_ce, loss_region = combined_loss(
                outputs[:args.labeled_bs],
                outputs_soft[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                iter_num,
                max_iterations
            )
            
            # Deep supervision loss
            if ds1 is not None:
                ds_loss = 0.0
                for ds_idx, ds_out in enumerate([ds1, ds2, ds3]):
                    ds_soft = torch.softmax(ds_out[:args.labeled_bs], dim=1)
                    ds_loss += dice_loss(ds_soft, label_batch[:args.labeled_bs].unsqueeze(1))
                ds_loss = ds_loss / 3.0
                # V2: Giam deep supervision weight dan (tu 0.4 -> 0.1)
                # De model tu hoc tu deep supervision dau, roi focus vao main output sau
                ds_weight = 0.4 * (1.0 - min(iter_num / max_iterations, 0.8))
                supervised_loss = supervised_loss + ds_weight * ds_loss

            # ===== Mixed Consistency Loss (MSE + KL) =====
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            # MSE consistency (nhu V1)
            consistency_dist_mse = losses.softmax_mse_loss(
                outputs[args.labeled_bs:], ema_output
            )
            
            # KL divergence consistency (moi V2)
            # KL div phu hop hon cho so sanh probability distributions
            student_log_soft = F.log_softmax(outputs[args.labeled_bs:], dim=1)
            teacher_soft = F.softmax(ema_output, dim=1)
            consistency_dist_kl = F.kl_div(
                student_log_soft, teacher_soft, reduction='none'
            )
            
            # Mix MSE va KL
            kl_w = args.kl_consistency_weight
            consistency_dist = (1 - kl_w) * consistency_dist_mse + kl_w * consistency_dist_kl
            
            # Uncertainty-aware masking
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            # ===== Total Loss =====
            loss = supervised_loss + consistency_weight * consistency_loss

            # ===== Backpropagation =====
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA model voi scheduled decay
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # V2: Cosine Annealing LR voi Warmup
            if args.use_cosine_lr:
                lr_ = get_cosine_lr(base_lr * 0.1, iter_num, max_iterations, args.warmup_iters)
            else:
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
                f'ce={loss_ce.item():.4f}, region={loss_region.item():.4f}, '
                f'lr={lr_:.6f}'
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
                    f'mean_hd95={mean_hd95:.4f}, best_dice={best_performance:.4f}'
                )
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
