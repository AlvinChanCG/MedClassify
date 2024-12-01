import torch
import models
import os
import random
from torchvision import transforms, datasets

MEANS = {'cifar': [0.4914, 0.4822, 0.4465],
         'imagenet': [0.485, 0.456, 0.406],
         'APTOS2019': [0.35872298, 0.2297728, 0.14966501]
        }
STDS = {'cifar': [0.2023, 0.1994, 0.2010],
        'imagenet': [0.229, 0.224, 0.225],
        'APTOS2019': [0.23028904, 0.15516128, 0.105367064]}

def get_model(args, **kwargs):

    if args.model == 'ConvNet':
        model = models.__dict__['ConvNet'](num_classes=args.nclasses,
                                           kernelsize=args.kernelsize,
                                           net_norm=args.norm_type,
                                           net_depth=args.net_depth,
                                           net_width=args.net_width,
                                           im_size=args.im_size,
                                           channel=args.datachannel,
                                           **kwargs)
    elif args.model == 'ResNetAP':
        model = models.__dict__['ResNetAP'](args=args)


    return model


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Grayscale(object):
    """
    NOTE: aiming at tensor aug
    """
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    """
    NOTE: aiming at tensor aug
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    """
    NOTE: aiming at tensor aug
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    """
    NOTE: aiming at tensor aug
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    """
    NOTE: aiming at tensor aug
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)

class Lighting(object):
    """
    NOTE: aiming at tensor aug
    Lighting noise(AlexNet - style PCA - based noise)
    """
    def __init__(self, alphastd, eigval, eigvec, device='cpu'):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


def transform(size=-1, augment=False, rrc=True, rrc_size=-1, normalize=True,):
    """
    Dataset transform
    Args:
        size (int): target size
        augment (bool):
        rrc (bool): Using RandomResizedCrop or not.
        rrc_size (int): target RRC size
        normalize (bool):

    Returns:

    """

    if size > 0:
        resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
        resize_test = [transforms.Resize(size), transforms.CenterCrop(size)]
        # print(f"Resize and crop training images to {size}")
    elif size == 0:
        resize_train = []
        resize_test = []
        assert rrc_size > 0, "Plz Set RRC size!"
    # else:
    #     # default set 224
    #     resize_train = [transforms.RandomResizedCrop(224)]
    #     resize_test = [transforms.Resize(256), transforms.CenterCrop(224)]

    if not augment:
        aug = []
        # print("Loader with DSA augmentation")
    else:
        jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[
                                      [-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203],
                                  ])
        aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

        if rrc and size >= 0:
            if rrc_size == -1:
                rrc_size = size
            rrc_fn = transforms.RandomResizedCrop(rrc_size, scale=(0.5, 1.0))
            aug = [rrc_fn] + aug
            print("Dataset with basic augmentation and RRC")
        else:
            print("Dataset with basic augmentation")


    cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(resize_train + cast + aug + normal_fn)
    test_transform = transforms.Compose(resize_test + cast + normal_fn)

    return train_transform, test_transform



def get_train_dataset(args, **kwargs):
    traindataroot = args.traindataroot
    size = args.im_size


    # train data
    if traindataroot and os.path.exists(traindataroot):
        print(f"traindataroot: {traindataroot}")
        # train_transforms = transforms.Compose([transforms.Resize(size),
        #                                        transforms.ToTensor(),
        #                                        transforms.Normalize()])
        traindataset = datasets.ImageFolder(traindataroot, transform=kwargs['train_transform'])

        return traindataset

    raise FileNotFoundError


def get_test_dataset(args, **kwargs):
    testdataroot = args.testdataroot

    size = args.im_size

    # test data
    if testdataroot and os.path.exists(testdataroot):
        print(f"testdataroot: {testdataroot}")
        # test_transforms = transforms.Compose([transforms.Resize(size),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize([0.35667878, 0.2302266, 0.14908062],
        #                                                            [0.22803271, 0.15399377, 0.10381984])])


        testdataset = datasets.ImageFolder(testdataroot, transform=kwargs['test_transform'])

        return testdataset

    raise FileNotFoundError


def get_val_dataset(args, **kwargs):
    valdataroot = args.valdataroot
    size = args.im_size

    # val data
    if valdataroot and os.path.exists(valdataroot):
        print(f"valdataroot: {valdataroot}")
        val_transforms = transforms.Compose([transforms.Resize(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.35806048, 0.22751328, 0.14604147],
                                                                  [0.2311621, 0.15450972, 0.10365705])])
        valdataset = datasets.ImageFolder(valdataroot, transform=val_transforms)

        return valdataset

    raise FileNotFoundError





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count