import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from torch.utils import data
from torchvision.transforms import functional as F

import utils.transform as transform

num_classes = 7
ST_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

# landsat
# num_classes = 5
# ST_CLASSES = ['0: No change', '1: Farmland', '2: Desert', '3: Building', '4: Water']
# ST_COLORMAP = [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]]

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A = np.array([48.30, 46.27, 48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B = np.array([49.41, 47.01, 47.94])

root = './DATA_ROOT/'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im


def tensor2int(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = im * STD_A + MEAN_A
    else:
        im = im * STD_B + MEAN_B
    return im.astype(np.uint8)


def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def read_RSimages(mode):
    # assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    # label_A_dir = os.path.join(root, mode, 'label1_rgb')
    # label_B_dir = os.path.join(root, mode, 'label2_rgb')

    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for idx, it in enumerate(data_list):
        # print(it)
        if (it[-4:] == '.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)

            # print(img_B_path)
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)

            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            labels_A.append(label_A)
            labels_B.append(label_B)
        if not idx % 500 or idx == len(data_list): print('%d/%d images loaded.' % (idx, len(data_list)))
        # if idx>50: break

    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, imgs_list_B, labels_A, labels_B


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False, random_swap=False):
        self.mode = mode
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_id = os.path.basename(self.imgs_list_A[idx])
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        if self.mode == 'train' and self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD(img_A, img_B, label_A, label_B)
        if self.mode == 'train' and self.random_swap:
            img_A, img_B, label_A, label_B = transform.rand_swap_SCD(img_A, img_B, label_A, label_B)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B), img_id

    def __len__(self):
        return len(self.imgs_list_A)


class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        # imgA_dir = os.path.join(test_dir, 'im1')
        # imgB_dir = os.path.join(test_dir, 'im2')
        imgA_dir = os.path.join(test_dir)
        imgB_dir = os.path.join(test_dir)
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:] == '.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len
