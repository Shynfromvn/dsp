"""
test_2D_improved_v2.py - Test script cho ImprovedUNetV3 tu re_code_v2.py

Su dung: python test_2D_improved_v2.py --root_path ../data/ACDC --exp ACDC/UAMT_V3_1 --num_classes 4 --labeled_num 7

Model architecture khop voi re_code_v2.py:
- ImprovedUNetV3 voi EncoderV3 + DecoderV3
- SE blocks, Attention Gates, ASPP, GroupNorm
- Wider channels [32, 64, 128, 256, 512]
- TTA (Test-Time Augmentation) de tang accuracy
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/UAMT_V3', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--use_tta', type=int, default=1,
                    help='Su dung Test-Time Augmentation (0/1)')


# =====================================================================
#     MODEL ARCHITECTURE V3 (copy CHINH XAC tu re_code_v2.py)
# =====================================================================

class SqueezeExcitation(nn.Module):
    """SE Block: Channel recalibration"""
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
    """Attention Gate cho skip connections"""
    def __init__(self, F_g, F_l, F_int):
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
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(self.W_g(g_up) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
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
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        out = torch.cat([
            self.conv1(x), self.conv6(x), self.conv12(x),
            F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        ], dim=1)
        return self.fuse(out)


class ResConvBlock(nn.Module):
    """Residual Conv Block voi SE + GroupNorm"""
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
    """Encoder voi wider channels + SE blocks"""
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
    """Decoder voi Attention Gates + ASPP + Deep Supervision"""
    def __init__(self, params):
        super().__init__()
        ft = params['feature_chns']
        self.n_class = params['class_num']
        
        self.aspp = ASPP(ft[4], ft[4])
        
        self.ag3 = AttentionGate(ft[4], ft[3], ft[3] // 2)
        self.ag2 = AttentionGate(ft[3], ft[2], ft[2] // 2)
        self.ag1 = AttentionGate(ft[2], ft[1], ft[1] // 2)
        self.ag0 = AttentionGate(ft[1], ft[0], ft[0] // 2)
        
        self.up4 = nn.ConvTranspose2d(ft[4], ft[3], kernel_size=2, stride=2)
        self.conv4 = ResConvBlock(ft[3] * 2, ft[3], 0.0, use_se=True)
        
        self.up3 = nn.ConvTranspose2d(ft[3], ft[2], kernel_size=2, stride=2)
        self.conv3 = ResConvBlock(ft[2] * 2, ft[2], 0.0, use_se=True)
        
        self.up2 = nn.ConvTranspose2d(ft[2], ft[1], kernel_size=2, stride=2)
        self.conv2 = ResConvBlock(ft[1] * 2, ft[1], 0.0, use_se=True)
        
        self.up1 = nn.ConvTranspose2d(ft[1], ft[0], kernel_size=2, stride=2)
        self.conv1 = ResConvBlock(ft[0] * 2, ft[0], 0.0, use_se=True)
        
        self.out_conv = nn.Conv2d(ft[0], self.n_class, kernel_size=1)
        
        self.ds_out3 = nn.Conv2d(ft[3], self.n_class, kernel_size=1)
        self.ds_out2 = nn.Conv2d(ft[2], self.n_class, kernel_size=1)
        self.ds_out1 = nn.Conv2d(ft[1], self.n_class, kernel_size=1)
    
    def forward(self, features, return_deep_supervision=False):
        x0, x1, x2, x3, x4 = features
        target_shape = x0.shape[2:]
        
        x4 = self.aspp(x4)
        
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
    ImprovedUNetV3 - khop voi re_code_v2.py
    - SE blocks, Attention Gates, ASPP, GroupNorm
    - Channels [32, 64, 128, 256, 512]
    - Dropout [0.05, 0.1, 0.15, 0.2, 0.3]
    """
    def __init__(self, in_chns, class_num, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        params = {
            'in_chns': in_chns,
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout': [0.05, 0.1, 0.15, 0.2, 0.3],
            'class_num': class_num,
        }
        
        self.encoder = EncoderV3(params)
        self.decoder = DecoderV3(params)
        self._init_weights()
    
    def _init_weights(self):
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
#                    TEST FUNCTIONS
# =====================================================================

def calculate_metric_percase(pred, gt):
    """Tinh Dice, HD95, ASD cho mot case"""
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd95, asd


def test_single_volume_tta(case, net, test_save_path, FLAGS):
    """
    Test-Time Augmentation (TTA): flip augmentations de tang accuracy
    """
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        slice_resized = zoom(slice_img, (256 / x, 256 / y), order=0)
        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            if FLAGS.use_tta:
                probs_list = []
                
                # 1. Original
                out = net(input_tensor)
                if isinstance(out, tuple):
                    out = out[0]
                probs_list.append(torch.softmax(out, dim=1))
                
                # 2. Horizontal flip
                flipped_h = torch.flip(input_tensor, [3])
                out_h = net(flipped_h)
                if isinstance(out_h, tuple):
                    out_h = out_h[0]
                probs_list.append(torch.flip(torch.softmax(out_h, dim=1), [3]))
                
                # 3. Vertical flip
                flipped_v = torch.flip(input_tensor, [2])
                out_v = net(flipped_v)
                if isinstance(out_v, tuple):
                    out_v = out_v[0]
                probs_list.append(torch.flip(torch.softmax(out_v, dim=1), [2]))
                
                # 4. Both flips
                flipped_hv = torch.flip(input_tensor, [2, 3])
                out_hv = net(flipped_hv)
                if isinstance(out_hv, tuple):
                    out_hv = out_hv[0]
                probs_list.append(torch.flip(torch.softmax(out_hv, dim=1), [2, 3]))
                
                avg_probs = torch.mean(torch.stack(probs_list), dim=0)
                out = torch.argmax(avg_probs, dim=1).squeeze(0)
            else:
                out_main = net(input_tensor)
                if isinstance(out_main, tuple):
                    out_main = out_main[0]
                out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    # Su dung ImprovedUNetV3 (khop voi re_code_v2.py)
    net = ImprovedUNetV3(in_chns=1, class_num=FLAGS.num_classes, deep_supervision=False)
    net = net.cuda()
    
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    state_dict = torch.load(save_mode_path)
    net.load_state_dict(state_dict, strict=False)
    print("init weight from {}".format(save_mode_path))
    print("TTA enabled: {}".format(bool(FLAGS.use_tta)))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume_tta(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    
    avg_metric = [
        first_total / len(image_list),
        second_total / len(image_list),
        third_total / len(image_list)
    ]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print("=== Results per class (Dice, HD95, ASD) ===")
    print(f"RV:  Dice={metric[0][0]:.4f}, HD95={metric[0][1]:.2f}, ASD={metric[0][2]:.2f}")
    print(f"MYO: Dice={metric[1][0]:.4f}, HD95={metric[1][1]:.2f}, ASD={metric[1][2]:.2f}")
    print(f"LV:  Dice={metric[2][0]:.4f}, HD95={metric[2][1]:.2f}, ASD={metric[2][2]:.2f}")
    avg = (metric[0] + metric[1] + metric[2]) / 3
    print(f"=== Average: Dice={avg[0]:.4f}, HD95={avg[1]:.2f}, ASD={avg[2]:.2f} ===")
