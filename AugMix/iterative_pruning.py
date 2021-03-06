# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.
Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.
Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

import augmentations
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

import models.cifar.resnet as res

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='resnet34')

# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
    
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--init', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')
                    
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
parser.add_argument('--steps', type=int, default=10, help='pruning steps')
parser.add_argument('--percent', default=0.2, type=float)

# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

parser.add_argument('--data_dir', default='../../data', type=str)
parser.add_argument('--c_data_dir', default='../../data/CIFAR-10-C/', type=str)

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    if args.no_jsd:
      images = images.cuda()
      targets = targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
    else:
      images_all = torch.cat(images, 0).cuda()
      targets = targets.cuda()
      logits_all = net(images_all)
      logits_clean, logits_aug1, logits_aug2 = torch.split(
          logits_all, images[0].size(0))

      # Cross-entropy is only computed on clean images
      loss = F.cross_entropy(logits_clean, targets)

      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)

      # Clamp mixture distribution to avoid exploding KL divergence
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss.backward()

    for k, m in enumerate(net.modules()):
        # print(k, m)
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m.weight.grad.data.mul_(mask)

    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total

def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])
  test_transform = preprocess

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        args.data_dir, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        args.data_dir, train=False, transform=test_transform, download=True)
    base_c_path = args.c_data_dir
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = datasets.CIFAR100(
        args.data_dir, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        args.data_dir, train=False, transform=test_transform, download=True)
    base_c_path = args.c_data_dir
    num_classes = 100
  # elif args.dataset == 'mnist':
  #   mean = (0.1307,)
  #   std = (0.3081,)
  #   transform_train = transforms.Compose([
  #       transforms.ToTensor(),
  #       transforms.Normalize(mean, std),
  #   ])

  #   transform_test = transforms.Compose([
  #       transforms.ToTensor(),
  #       transforms.Normalize(mean, std),
  #   ])
  #   train_data = dataloader(root='../data', train=True, download=True, transform=transform_train)
  #   test_data = dataloader(root='../data', train=False, download=False, transform=transform_test)

  #   num_classes = 10

  train_data = AugMixDataset(train_data, preprocess, args.no_jsd)
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  #=============== Create model ===============
  if args.model =='resnet34':
    net = res.ResNet34(num_classes=num_classes)
    net_ref = res.ResNet34(num_classes=num_classes)

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay,
      nesterov=True)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  net_ref = torch.nn.DataParallel(net_ref).cuda()
  # net = net.cuda()
  # net_ref = net_ref.cuda()
  cudnn.benchmark = True

  start_epoch = 0

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))

  # scheduler = MultiStepLR(optimizer,
  #                         milestones=[80, 120],
  #                         gamma=0.1)

  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      net_ref.load_state_dict(checkpoint['state_dict'])
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  # set some weights to zero, according to model_ref ---------------------------------
  if args.init:
    checkpoint = torch.load(args.init)
    net.load_state_dict(checkpoint['state_dict'])


  for stp in range(args.steps):
    print('Step : ' + str(stp))

    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss0, test_acc0 = test(net_ref, test_loader)
    print('before pruning : Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss0, 100 - 100. * test_acc0))
    # -------------------------------------------------------------
    #pruning
    total = 0
    total_nonzero = 0
    for m in net_ref.modules():
      if isinstance(m, nn.Conv2d):
        total += m.weight.data.numel()
        mask = m.weight.data.abs().clone().gt(0).float().cuda()
        total_nonzero += torch.sum(mask)

    conv_weights = torch.zeros(total)
    index = 0
    for m in net_ref.modules():
      if isinstance(m, nn.Conv2d):
        size = m.weight.data.numel()
        conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
        index += size

    y, i = torch.sort(conv_weights)
    # thre_index = int(total * args.percent)
    thre_index = total - total_nonzero + int(total_nonzero * args.percent)
    thre = y[int(thre_index)]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(net_ref.modules()):
      if isinstance(m, nn.Conv2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.numel() - torch.sum(mask)
        m.weight.data.mul_(mask)
        if int(torch.sum(mask)) == 0:
          zero_flag = True
          print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
            format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    # -------------------------------------------------------------
    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss1, test_acc1 = test(net_ref, test_loader)
    print('after pruning : Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss1, 100 - 100. * test_acc1))
    filepath = os.path.join(os.path.join(os.path.join(os.path.join(args.save, 'prune/'),args.model),args.dataset), str(stp+1)+'step')
    if not os.path.exists(filepath):
      os.makedirs(filepath)
    save_checkpoint({
            'epoch': 0,
            'state_dict': net_ref.state_dict(),
            'acc': test_acc1,
            'best_acc': 0.,
            # 'optimizer' : optimizer.state_dict(),
        }, False, checkpoint=filepath)

    with open(os.path.join(filepath, 'prune.txt'), 'w') as f:
        f.write('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss0, test_acc0))
        f.write('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
        f.write('After Pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss1, test_acc1))

        if zero_flag:
            f.write("There exists a layer with 0 parameters left.")

    # -------------- copy mask ---------------
    for m, m_ref in zip(net.modules(), net_ref.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m_ref.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m.weight.data.mul_(mask)

    filepath = os.path.join(os.path.join(os.path.join(os.path.join(args.save, 'retrain/'),args.model),args.dataset), str(stp+1)+'step')
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # --------------- Train and val -----------------
    log_path = os.path.join(filepath, args.dataset + '_' + args.model + '_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    best_acc = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        begin_time = time.time()

        train_loss_ema = train(net, train_loader, optimizer, scheduler)
        test_loss, test_acc = test(net, test_loader)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': args.model,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(filepath, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(filepath, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_time,
            train_loss_ema,
            test_loss,
            100 - 100. * test_acc,
        ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
            ' Test Error {4:.2f}'
            .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                    test_loss, 100 - 100. * test_acc))

    test_c_acc = test_c(net, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))
    
    # ====== initialize ======
    best_path = os.path.join(filepath, 'model_best.pth.tar') # model best vs checkpoint
    checkpoint = torch.load(best_path)
    net_ref.load_state_dict(checkpoint['state_dict'])

    checkpoint = torch.load(args.init)
    net.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

if __name__ == '__main__':
  main()