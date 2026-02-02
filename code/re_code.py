import argparse
import logging
import os
import random
import shutil
import sys
import time
import math

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
                    default='ACDC/Mean_Teacher', help='experiment_name')
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
args = parser.parse_args()


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
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# ============================================================================
# CẢI TIẾN #5: FOCAL LOSS - Xử lý class imbalance
# ============================================================================
# Focal Loss giảm trọng số của các mẫu dễ phân loại, tập trung vào mẫu khó
# Công thức: FL(p) = -alpha * (1-p)^gamma * log(p)
# - alpha: trọng số cho positive class (0.25)
# - gamma: focusing parameter (2) - gamma càng cao, càng tập trung vào mẫu khó
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================================
# CẢI TIẾN #4: STRONG AUGMENTATION - Tăng cường dữ liệu cho unlabeled data
# ============================================================================
# Augmentation mạnh giúp model học robust hơn
# Bao gồm: Gaussian noise + Random brightness + Random blur
# ============================================================================
def strong_augment(x):
    """
    Strong augmentation cho unlabeled data.
    Args:
        x: Input tensor (B, C, H, W)
    Returns:
        Augmented tensor
    """
    # Gaussian noise (luôn áp dụng)
    noise = torch.clamp(torch.randn_like(x) * 0.1, -0.2, 0.2)
    x = x + noise
    
    # Random brightness/contrast (50% chance)
    if random.random() > 0.5:
        brightness_factor = 0.8 + random.random() * 0.4  # 0.8 - 1.2
        x = x * brightness_factor
    
    # Random Gaussian blur (30% chance)
    if random.random() > 0.7:
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    
    return x


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    # ========================================================================
    # CẢI TIẾN #5: Khởi tạo các loss functions
    # ========================================================================
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    focal_loss = FocalLoss(alpha=0.25, gamma=2)  # Thêm Focal Loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            # ================================================================
            # CẢI TIẾN #4: Sử dụng Strong Augmentation thay vì chỉ noise
            # ================================================================
            ema_inputs = strong_augment(unlabeled_volume_batch)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            # ================================================================
            # CẢI TIẾN #5: Kết hợp CE + Dice + Focal Loss
            # ================================================================
            # Supervised loss trên labeled data
            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_focal = focal_loss(outputs[:args.labeled_bs],
                                    label_batch[:][:args.labeled_bs].long())
            
            # Kết hợp: 40% Dice + 30% CE + 30% Focal
            supervised_loss = 0.4 * loss_dice + 0.3 * loss_ce + 0.3 * loss_focal

            # ================================================================
            # CẢI TIẾN #1: UNCERTAINTY ESTIMATION (Quan trọng nhất!)
            # ================================================================
            # Sử dụng Monte Carlo sampling để ước lượng uncertainty
            # Chỉ enforce consistency trên vùng có độ tin cậy cao (uncertainty thấp)
            # ================================================================
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                # Monte Carlo sampling để ước lượng uncertainty
                T = 8  # Số lần sampling
                _, _, w, h = unlabeled_volume_batch.shape
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
                
                # Chạy T/2 lần với noise khác nhau để có T predictions
                for i in range(T // 2):
                    ema_inputs_mc = volume_batch_r + \
                        torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs_mc)
                
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, num_classes, w, h)
                preds = torch.mean(preds, dim=0)  # Trung bình T predictions
                
                # Tính uncertainty bằng entropy
                # Entropy cao = uncertainty cao = không tin tưởng
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
                
                # Adaptive threshold: tăng dần theo thời gian
                # Ban đầu: threshold cao (filter nhiều vùng không chắc chắn)
                # Về sau: threshold thấp (sử dụng nhiều vùng hơn)
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                mask = (uncertainty < threshold).float()
                
                # Consistency loss chỉ tính trên vùng có độ tin cậy cao
                consistency_dist = (outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2
                consistency_loss = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

            loss = supervised_loss + consistency_weight * consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # ================================================================
            # CẢI TIẾN #2: COSINE ANNEALING LEARNING RATE
            # ================================================================
            # Cosine annealing thường cho kết quả tốt hơn polynomial decay
            # LR giảm mượt theo hàm cosine từ base_lr xuống 0
            # ================================================================
            
            # Warm-up trong 1000 iterations đầu tiên
            warmup_iters = 1000
            if iter_num < warmup_iters:
                # Linear warm-up: LR tăng dần từ 0 lên base_lr
                lr_ = base_lr * iter_num / warmup_iters
            else:
                # Cosine annealing: LR giảm dần theo hàm cosine
                lr_ = base_lr * 0.5 * (1 + math.cos(math.pi * (iter_num - warmup_iters) / (max_iterations - warmup_iters)))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)  # Thêm log focal loss
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_focal: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_focal.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

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

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)