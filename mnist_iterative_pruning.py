from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np

import models.cifar.cnn2 as cnn2
import models.cifar.cnn4 as cnn4
import models.cifar.cnn6 as cnn6

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.misc import get_conv_zero_param


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='mnist', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[7, 9],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--init', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='conv4')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=2, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results_c/', type=str)
parser.add_argument('--data_dir', default='../data', type=str)
parser.add_argument('--c_data_dir', default='../data/mnist-c/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Pruning options
parser.add_argument('--percent', default=0.2, type=float)
parser.add_argument('--steps', type=int, default=10, help='pruning steps')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

CORRUPTIONS = ['identity',
                'shot_noise',
                'impulse_noise',
                'glass_blur',
                'motion_blur',
                'shear',
                'scale',
                'rotate',
                'brightness',
                'translate',
                'stripe',
                'fog',
                'spatter',
                'dotted_line',
                'zigzag',
                'canny_edges',]

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        base_c_path = args.c_data_dir
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        base_c_path = args.c_data_dir
    elif args.dataset == 'mnist':
        dataloader = datasets.MNIST
        num_classes = 10
        mean = (0.1307,)
        std = (0.3081,)
        base_c_path = args.c_data_dir
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('conv2'):
        print('conv2')
        model = cnn2.cnn()
        model_ref = cnn2.cnn()
    elif args.arch.endswith('conv4'):
        print('conv4')
        model = cnn4.cnn()
        model_ref = cnn4.cnn()
    elif args.arch.endswith('conv6'):
        print('conv6')
        model = cnn6.cnn()
        model_ref = cnn6.cnn()
    model.cuda()
    model_ref.cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = 0
        model_ref.load_state_dict(checkpoint['state_dict'])
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # set some weights to zero, according to model_ref ---------------------------------
    if args.init:
        print('==> Loading init model from %s'%args.init)
        checkpoint = torch.load(args.init)
        model.load_state_dict(checkpoint['state_dict'])

    # ============== 0step c-test ================
    mean_test_c_acc, test_c_errs = test_c(model_ref, testset, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * mean_test_c_acc))

    filepath = os.path.join(os.path.join(os.path.join(os.path.join(args.save_dir, 'train/'),args.arch),args.dataset))
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(os.path.join(filepath, 'c-test-result.txt'), 'w') as f:
        f.write('iid test error rate : {}'.format(100 - best_acc))
        f.write('\n')
        f.write('ood test error rate : {}'.format(100 - 100. * mean_test_c_acc))
        f.write('\n')
        f.write('ood list : {}'.format(test_c_errs))

    testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    # for loop -----------------------------------------------------
    for step in range(args.steps):
        print('Step : ' + str(step))
        # -------------------------------------------------------------
        #pruning
        print('\nEvaluation only')
        test_loss0, test_acc0 = test(testloader, model_ref, criterion, start_epoch, use_cuda)
        print('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss0, test_acc0))
 
        total = 0
        total_nonzero = 0
        for m in model_ref.modules():
            if isinstance(m, nn.Conv2d):
                total += m.weight.data.numel()
                mask = m.weight.data.abs().clone().gt(0).float().cuda()
                total_nonzero += torch.sum(mask)

        conv_weights = torch.zeros(total)
        index = 0
        for m in model_ref.modules():
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
        for k, m in enumerate(model_ref.modules()):
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

        print('\nTesting')
        test_loss1, test_acc1 = test(testloader, model_ref, criterion, start_epoch, use_cuda)
        print('After Pruning: Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss1, test_acc1))
        filepath = os.path.join(os.path.join(os.path.join(os.path.join(args.save_dir, 'prune/'),args.arch),args.dataset), str(step+1)+'step')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        save_checkpoint({
                'epoch': 0,
                'state_dict': model_ref.state_dict(),
                'acc': test_acc1,
                'best_acc': 0.,
            }, False, checkpoint=filepath)

        with open(os.path.join(filepath, 'prune.txt'), 'w') as f:
            f.write('Before pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss0, test_acc0))
            f.write('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
            f.write('After Pruning: Test Loss:  %.8f, Test Acc:  %.2f\n' % (test_loss1, test_acc1))

            if zero_flag:
                f.write("There exists a layer with 0 parameters left.")

        # -------------- copy mask ---------------
        for m, m_ref in zip(model.modules(), model_ref.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m_ref.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.data.mul_(mask)
        
        filepath = os.path.join(os.path.join(os.path.join(os.path.join(args.save_dir, 'retrained/'),args.arch),args.dataset), str(step+1)+'step')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        best_acc = 0
        logger = Logger(os.path.join(filepath, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

        # --------------- Train and val -----------------
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
            num_parameters = get_conv_zero_param(model)
            print('Zero parameters: {}'.format(num_parameters))
            num_parameters = sum([param.nelement() for param in model.parameters()])
            print('Parameters: {}'.format(num_parameters))

            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
            # scheduler.step()
            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            checkpoint = {
                'epoch': epoch,
                'dataset': args.dataset,
                'model': args.init,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }

            save_path = os.path.join(filepath, 'checkpoint.pth.tar')
            torch.save(checkpoint, save_path)
            if is_best:
                shutil.copyfile(save_path, os.path.join(filepath, 'model_best.pth.tar'))

        logger.close()
        
        best_path = os.path.join(filepath, 'model_best.pth.tar') # model best vs checkpoint
        checkpoint = torch.load(best_path)
        model_ref.load_state_dict(checkpoint['state_dict'])
        
        if args.init:
            checkpoint = torch.load(args.init)
            model.load_state_dict(checkpoint['state_dict'])
        else :
            model = cnn4.cnn()
            model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        state['lr'] = args.lr

        # ============== c-test ================
        mean_test_c_acc, test_c_errs = test_c(model_ref, testset, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * mean_test_c_acc))

        with open(os.path.join(filepath, 'c-test-result.txt'), 'w') as f:
            f.write('iid test error rate : {}'.format(100 - best_acc))
            f.write('\n')
            f.write('ood test error rate : {}'.format(100 - 100. * mean_test_c_acc))
            f.write('\n')
            f.write('ood list : {}'.format(test_c_errs))

        testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    return

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)
def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)
def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    corruption_errs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        datapath = os.path.join(base_path,corruption)
        test_data.data = torch.tensor(np.load(os.path.join(datapath, 'test_images.npy'))[:,:,:,0])
        test_data.targets = torch.LongTensor(np.load(os.path.join(datapath, 'test_labels.npy')))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

        test_loss, test_acc = test2(net, test_loader)
        corruption_accs.append(test_acc)
        corruption_errs.append(100 - 100. * test_acc)

        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))
    return np.mean(corruption_accs), corruption_errs
    
def test2(net, test_loader):
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
        
def save_checkpoint(state, is_best, checkpoint, filename='pruned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
