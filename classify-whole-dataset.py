"""
CUDA_VISIBLE_DEVICES=3 python classify-whole-dataset.py

 --traindataroot  ./datasets/APTOS2019/train --testdataroot ./datasets/APTOS2019/test  --train_bs 8 --test_bs 64  --im_size 256 256  --datachannel 3  --nclasses 5  --model ResNetAP  --net_depth 18  --norm_type batch

CUDA_VISIBLE_DEVICES=3 python classify-whole-dataset.py --traindataroot  ./datasets/APTOS2019/aptosbegin10w-ipc50_train --testdataroot ./datasets/APTOS2019/test  --use_balance --balance_method undersampling --train_bs 8 --test_bs 64  --im_size 256 256  --datachannel 3  --nclasses 5  --model ResNetAP  --net_depth 18  --norm_type instance  --epochs 1000  --use_basic_aug  --use_rrc_aug  --rrc_size 224  --use_mixup
"""
import os.path
import models
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, default_collate
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torchinfo
import argparse
import utils

torch.backends.cudnn.benchmark = True

# print(f"models.__dict__['ConvNet'] = {models.__dict__['ConvNet']}")


# traindataroot = r"./datasets/APTOS2019/train"  # r"D:\Datasets\EyePACS-AIROGS-light-v1\release-crop\train"
# testdataroot = r"./datasets/APTOS2019/test" # r"D:\Datasets\EyePACS-AIROGS-light-v2\test"  # r"D:\Datasets\EyePACS-AIROGS-light-v1\release-crop\test"
# valdataroot = r"./datasets/APTOS2019/validation"
#
#
# batchsize = 32  #64
# nclass = 5
# size = (32,32)   # (256, 256)
#
# # train data
# if traindataroot and os.path.exists(traindataroot):
#     print(f"traindataroot: {traindataroot}")
#     train_transforms = transforms.Compose([transforms.Resize(size),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.35872298, 0.2297728, 0.14966501],
#                                                               [0.23028904, 0.15516128, 0.105367064])])
#     traindataset = datasets.ImageFolder(traindataroot, transform=train_transforms)
#     train_loader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False, pin_memory=False)
#
# # test data
# if testdataroot and os.path.exists(testdataroot):
#     print(f"testdataroot: {testdataroot}")
#     test_transforms = transforms.Compose([transforms.Resize(size),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.35667878, 0.2302266, 0.14908062],
#                                                                 [0.22803271, 0.15399377, 0.10381984])])
#     testdataset = datasets.ImageFolder(testdataroot, transform=test_transforms)
#     test_loader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
#
# # val data
# if valdataroot and os.path.exists(valdataroot):
#     print(f"valdataroot: {valdataroot}")
#     val_transforms = transforms.Compose([transforms.Resize(size),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.35806048, 0.22751328, 0.14604147],
#                                                                 [0.2311621, 0.15450972, 0.10365705])])
#     valdataset = datasets.ImageFolder(valdataroot, transform=val_transforms)
#     val_loader = DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)


# model define
"""
class ConvNet(nn.Module):
    def __init__(self,
                 num_classes,
                 net_norm='instance',
                 net_depth=3,
                 net_width=128,
                 channel=3,
                 net_act='relu',
                 net_pooling='avgpooling',
                 im_size=(32, 32),
                 kernelsize=3):
        print(f"Define Convnet (depth {net_depth}, width {net_width}, norm {net_norm}, im_size {im_size})")
        super(ConvNet, self).__init__()
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU()
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

        if net_pooling == 'maxpooling':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

        self.depth = net_depth
        self.net_norm = net_norm

        self.layers, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm,
                                                    net_pooling, im_size, kernelsize)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        # print(f"--> shape_feat: {shape_feat}")
        # print(f"==> Shape of feat: {shape_feat}")                          # [128, 28, 28]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        # print(f"==> x shape: {x.shape}")
        for d in range(self.depth):
            # print(f"here: {next(self.layers['conv'][d].parameters()).device}")
            x = self.layers['conv'][d](x)
            if len(self.layers['norm']) > 0:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if len(self.layers['pool']) > 0:
                x = self.layers['pool'][d](x)

        # print(f'==> Here x shape: {x.shape}')
        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)

        if return_features:
            return logit, out
        else:
            return logit

    def embed(self, x):
        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if len(self.layers['norm']) > 0:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if len(self.layers['pool']) > 0:
                x = self.layers['pool'][d](x)

        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)  # (128, 4, 4)

        return out

    def embed_to_logit(self, x):
        logit = self.classifier(x)
        return logit

    def get_feature(self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False):
        if idx_to == -1:
            idx_to = idx_from
        features = []

        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if self.net_norm:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if self.net_pooling:
                x = self.layers['pool'][d](x)
            features.append(x)
            # print("length:", len(features))
            if idx_to < len(features):
                return features[idx_from:idx_to + 1]

        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from:idx_to + 1]

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c * h * w)
        if net_norm == 'batch':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layer':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instance':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'group':
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s' % net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size, kernelsize):
        layers = {'conv': [], 'norm': [], 'act': [], 'pool': []}

        # print(f"_make_layers params including: channel:{channel}, net_width:{net_width}, net_depth:{net_depth}, im_size:{im_size}")
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]  # 参数默认是[3,224,224], 可能是由于argument.py的args.size没改
        # print(f"-> Ori shape_feat: {shape_feat}")

        for d in range(net_depth):
            # layers['conv'] += [
            #     nn.Conv2d(in_channels,
            #               net_width,
            #               kernel_size=kernelsize,        # 3,
            #               padding=3 if channel == 1 and d == 0 else 1)
            # ]

            if channel == 1 and d == 0:
                padding = 3

            elif channel!=1 and kernelsize==3:
                padding = 1
            elif channel != 1 and kernelsize == 5:
                padding = 2
            elif channel != 1 and kernelsize == 7:
                padding = 3
            elif channel != 1 and kernelsize == 9:
                padding = 4

            layers['conv'] += [
                nn.Conv2d(in_channels,
                          net_width,
                          kernel_size=kernelsize,  # 3,
                          padding=padding)
            ]

            # print(f"new conv here: {next(layers['conv'][d].parameters()).device}")
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers['norm'] += [self._get_normlayer(net_norm, shape_feat)]
            # print(f"new norm here: {next(layers['norm'][d].parameters()).device}")
            layers['act'] += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers['pool'] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        layers['conv'] = nn.ModuleList(layers['conv'])
        layers['norm'] = nn.ModuleList(layers['norm'])
        layers['act'] = nn.ModuleList(layers['act'])
        layers['pool'] = nn.ModuleList(layers['pool'])
        layers = nn.ModuleDict(layers)

        # print(f"new new conv: {next(layers['conv'][0].parameters()).device}")

        return layers, shape_feat
"""

def train(model, device, train_loader, optimizer, epoch):
    pass


# def evaluate_top1(model, device, loader, epoch):
#     model.eval()
#     total_correct_num = 0
#     total_num = 0
#     for batch_idx, (data, target) in enumerate(loader):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         # current_batch_acc1, current_batch_correct_num = accuracy(output.data, target, topk=(1, ))
#         # # print(current_batch_correct_num)
#         # total_correct_num += current_batch_correct_num[0]
#         # total_num += data.size(0)
#
#         _, predicted = torch.max(output.data, 1)
#         total_correct_num += (predicted == target).sum().item()
#         total_num += target.size(0)
#
#     print(f"total_correct_num: {total_correct_num}, total_num: {total_num}")
#     acc = 100 * total_correct_num / total_num
#
#
#     return acc


def evaluate_top1(model, device, loader, criterion, epo):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    predicted_labels = []
    true_labels = []

    model.eval()
    for batch_idx, (inpdata, target) in enumerate(loader):
        inpdata, target = inpdata.to(device), target.to(device)
        out = model(inpdata)

        loss = criterion(out, target)

        acc = accuracy(out.data, target, topk=(1,))
        top1acc = acc[0]

        # record
        losses.update(loss.item(), inpdata.size(0))
        top1.update(top1acc.item(), inpdata.size(0))

        predicted_labels.extend(torch.max(out.data, 1)[-1].tolist())
        true_labels.extend(target.tolist())

    cm = confusion_matrix(true_labels, predicted_labels)

    return top1, cm


def soft_accuracy(output, target, topk=(1,)):
    """Computes the soft accuracy over the k top predictions for soft labels."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Convert logits to probabilities using softmax
        probs = torch.softmax(output, dim=1)

        # Get top-k predictions indices
        _, pred = probs.topk(maxk, dim=1, largest=True, sorted=True)

        res = []
        for k in topk:
            # For each sample, gather the target probabilities corresponding to the top-k predictions
            top_k_probs = target.gather(1, pred[:, :k])

            # Sum the probabilities in the soft target for the top-k predictions, and then average over the batch
            correct_k = top_k_probs.sum(1).mean()
            res.append(correct_k * 100.0)  # Convert to percentage

        return res

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def custom_collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


if __name__=='__main__':

    # print(f"traindataroot: {traindataroot}")
    # print(f"testdataroot: {testdataroot}")


    parser = argparse.ArgumentParser(description='Params Process')
    parser.add_argument('--traindataroot', type=str, default='./datasets/APTOS2019/train')
    parser.add_argument('--testdataroot', type=str, default='./datasets/APTOS2019/test')
    parser.add_argument('--valdataroot', type=str, default='./datasets/APTOS2019/validation')
    # dataset balance setting
    parser.add_argument('--use_balance', action='store_true', help='use balanced dataset or not')
    parser.add_argument('--balance_method', type=str, default='None', choices=['None', 'undersampling'])
    parser.add_argument('--get_balance_by_dynamics', action='store_true', help='get balance by dynamics or static')
    # parser.add_argument('--balance_testset_root', type=str, default='./datasets/APTOS2019/balance_testset')
    #
    parser.add_argument('--train_bs', type=int, default=8, help='train batch size')
    parser.add_argument('--test_bs', type=int, default=64, help='test batch size')
    parser.add_argument('--im_size', type=int, nargs="+", default=[256, 256])  # '+' == 1 or more; '*' == 0 or more; '?' == 0 or 1.
    parser.add_argument('--datachannel', type=int, default=3)
    parser.add_argument('--nclasses', type=int, default=5)
    parser.add_argument('--model', type=str, default='ResNetAP')
    parser.add_argument('--kernelsize', type=int, default=3, help='convnet kernel size')
    parser.add_argument('--net_depth', type=int, default=3)
    parser.add_argument('--net_width', type=int, default=128)
    parser.add_argument('--width_factor', type=float, default=1.0, nargs="*", help="for resnetap")
    parser.add_argument('--norm_type', type=str, default='instance', choices=['batch', 'instance'])
    # training setting
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_basic_aug', action='store_true')
    parser.add_argument('--use_rrc_aug', action='store_true')
    parser.add_argument('--rrc_size', type=int, default=-1, help="-1 denotes using the same size as im_size")
    parser.add_argument('--use_mixup', action='store_true')
    #

    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # kernelsize = 3
    # net_width = 128

    # dataset transform setting
    train_transform, test_transform = utils.transform(size=args.im_size[0], augment=args.use_basic_aug, rrc=args.use_rrc_aug, rrc_size=args.rrc_size, normalize=True)


    # get datasets
    traindataset = utils.get_train_dataset(args, train_transform=train_transform)
    if args.use_balance:
        if args.get_balance_by_dynamics:
            print(f'Using dynamics balance...')
            testdataset = utils.get_test_dataset(args, test_transform=test_transform, use_balance=args.use_balance, balance_method=args.balance_method)
        else:
            print(f"Using static balance...")
            args.balance_testset_root = os.path.join(os.path.split(args.testdataroot)[0],
                                                     f"{args.balance_method}_balance_test")
            testdataset = datasets.ImageFolder(args.balance_testset_root, transform=test_transform)
    print(f"train dataset size: {len(traindataset)}, test dataset size: {len(testdataset)}")

    # mixup aug setting
    cutmix = v2.CutMix(num_classes=args.nclasses)
    mixup = v2.MixUp(num_classes=args.nclasses)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


    # get loader
    if args.use_mixup:
        train_loader = DataLoader(traindataset, batch_size=args.train_bs, shuffle=True, num_workers=4, drop_last=False, pin_memory=True, collate_fn=custom_collate_fn)
    else:
        train_loader = DataLoader(traindataset, batch_size=args.train_bs, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

    test_loader = DataLoader(testdataset, batch_size=args.test_bs, shuffle=False, num_workers=1, drop_last=False, pin_memory=False)

    # val_loader = DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)


    # get model
    # model = models.__dict__['ConvNet'](num_classes=nclass, im_size=size, net_width=net_width, kernelsize=kernelsize)
    model = utils.get_model(args)
    model.to(device)
    # print(f"Check whole：{next(model.parameters()).device}")
    # print(f"Check local：{next(model.layers['conv'].parameters()).device}")

    torchinfo.summary(model, input_size=(1, 3, *args.im_size), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)


    losses = []
    early_stop_limit = 20
    early_stop_count = 0
    best_acc = 0.0
    for epoch in range(args.epochs):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            # print(images.device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # print(f"前向。。。")
            # print(f"查看：{next(model.layers['conv'].parameters()).device}")
            outputs = model(images)
            loss = criterion(outputs, labels)

            # record loss, acc
            losses.update(loss.item(), images.size(0))
            if args.use_mixup:
                trainacc = soft_accuracy(outputs, labels, topk=(1,))
                traintop1 = trainacc[0]
            else:
                trainacc = accuracy(outputs.data, labels, topk=(1,))
                traintop1 = trainacc[0]
            top1.update(traintop1.item(), images.size(0))
            # top5.update(traintop5.item(), images.size(0))

            loss.backward()
            optimizer.step()

            print('| Epoch: %d, iter: %d, train Loss (avg): %.4f, train top1 (avg): %.4f, | current best eval top1: %.4f' % (epoch, i, losses.avg, top1.avg, best_acc))

        if epoch % 5 == 0:
            # testacc = evaluate_top1(model, device, test_loader, epoch)
            testtop1, cm = evaluate_top1(model, device, test_loader, criterion, epoch)
            print(f'==> (Evaluate) Epoch: {epoch}; Top1 (avg): {testtop1.avg:.4f};\n Confusion matrix:\n {cm}')
            if testtop1.avg >= best_acc:
                best_acc = testtop1.avg

                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count == early_stop_limit:
                    print(f"Early stop at {epoch} ep, best top1: {best_acc}")
                    break

        scheduler.step()

    print(f"Best Top-1 Acc: {best_acc}")
