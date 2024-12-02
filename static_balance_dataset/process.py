import torch
import numpy as np
import random
import os
import shutil
import argparse
from collections import defaultdict as ddict
from torchvision import datasets

def get_static_balance_dataset(args, **kwargs):
    dataset = datasets.ImageFolder(args.dataroot)

    if kwargs['balance_method'] == 'undersampling':
        d = ddict(list)
        for img_path, img_lab in dataset.imgs:
            d[img_lab].append(img_path)

        min_num = min([len(item) for item in d.values()])
        new_d = dict()
        for key in d:
            new_d[key] = np.random.choice(d[key], min_num, replace=False).tolist()



        cnt = 0
        for key in new_d:
            new_root = os.path.join(args.balance_dataroot, str(key))
            if not os.path.exists(new_root):
                os.makedirs(new_root)

            for file_path in new_d[key]:
                shutil.copy(file_path, new_root)
                cnt += 1
        print(f"new file cnt: {cnt}")



parser = argparse.ArgumentParser(description='param')
parser.add_argument('--dataroot', type=str)
parser.add_argument('--balance_dataroot', type=str, default='balance_dataset')
parser.add_argument('--balance_method', type=str, default='undersampling')

args = parser.parse_args()

get_static_balance_dataset(args, balance_method=args.balance_method)
print(f"Done!")


