"""
test_2D_improved.py - Test script cho ImprovedUNet từ re_code.py

Sử dụng: python test_2D_improved.py --root_path ../data/ACDC --exp ACDC/Uncertainty_Aware_Mean_Teacher_ffc_1 --num_classes 4 --labeled_num 7
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

# Import ImprovedUNet từ re_code (copy các class cần thiết)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Improved_UAMT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')


# =====================================================================
#                    MODEL ARCHITECTURE (copy từ re_code.py)
# =====================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
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
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvBlockWithAttention(nn.Module):
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
        return self.activation(out + residual)


class ImprovedEncoder(nn.Module):
    def __init__(self, params):
        super(ImprovedEncoder, self).__init__()
        self.in_chns = params['in_chns']
        self.ft_chns = params['feature_chns']
        self.dropout = params['dropout']
        
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
    def __init__(self, params):
        super(ImprovedDecoder, self).__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']
        
        self.up1 = self._make_up_block(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3])
        self.up2 = self._make_up_block(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2])
        self.up3 = self._make_up_block(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1])
        self.up4 = self._make_up_block(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0])
        
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
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
        
        x = self.up1['up'](x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1['conv'](x)
        if return_deep_supervision:
            ds3 = F.interpolate(self.ds_out3(x), size=target_shape, mode='bilinear', align_corners=True)
        
        x = self.up2['up'](x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2['conv'](x)
        if return_deep_supervision:
            ds2 = F.interpolate(self.ds_out2(x), size=target_shape, mode='bilinear', align_corners=True)
        
        x = self.up3['up'](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3['conv'](x)
        if return_deep_supervision:
            ds1 = F.interpolate(self.ds_out1(x), size=target_shape, mode='bilinear', align_corners=True)
        
        x = self.up4['up'](x)
        x = torch.cat([x, x0], dim=1)
        x = self.up4['conv'](x)
        
        output = self.out_conv(x)
        
        if return_deep_supervision:
            return output, ds1, ds2, ds3
        return output


class ImprovedUNet(nn.Module):
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
#                    TEST FUNCTIONS
# =====================================================================

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            # ImprovedUNet trả về single output khi eval
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
    
    # Sử dụng ImprovedUNet thay vì UNet gốc
    net = ImprovedUNet(in_chns=1, class_num=FLAGS.num_classes, deep_supervision=False)
    net = net.cuda()
    
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    # Load state dict với strict=False để bỏ qua deep supervision layers nếu có
    state_dict = torch.load(save_mode_path)
    net.load_state_dict(state_dict, strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
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
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
