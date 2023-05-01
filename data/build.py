# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets
from torchvision import transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler
from .dataset import *
from torch.utils.data import DataLoader, DistributedSampler
import numbers
from timm.data.transforms import str_to_pil_interp
from torch.utils.data._utils.collate import default_collate

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


class RandomSwap(object):
    def __init__(self, shuffle_patch_size):
        if isinstance(shuffle_patch_size, numbers.Number):
            self.shuffle_patch_size = (int(shuffle_patch_size), int(shuffle_patch_size))
        else:
            assert len(shuffle_patch_size) == 2, "Please provide only two dimensions (h, w) for size."
            self.shuffle_patch_size = shuffle_patch_size

    def crop_image(self, image):
        # numpy will change the high width position when storing the image.
        high, width = image.shape[1], image.shape[2]
        assert width%self.shuffle_patch_size[0]==0 and high%self.shuffle_patch_size[1]==0
        high_num, width_num = self.shuffle_patch_size[1], self.shuffle_patch_size[0]
        high_part, width_part = high//high_num, width//width_num
        crop_x = [width_part * i for i in range(width_num + 1)]
        crop_y = [high_part * i for i in range(high_num + 1)]
        im_list = []
        for i in range(len(crop_y) - 1):
            for j in range(len(crop_x) - 1):
                im_list.append(image[:, crop_y[i]:min(crop_y[i + 1], high), crop_x[j]:min(crop_x[j + 1], width)])
        return im_list,high_num,width_num

    def __call__(self, img):
        #print(type(img), img.shape)
        img_list, high_num, width_num = self.crop_image(img)
        permutation = np.random.permutation(len(img_list))
        img_list = [img_list[idx] for idx in permutation]
        img=torch.cat([torch.cat(img_list[i*width_num:(i+1)*width_num],dim=2) for i in range(high_num)],dim=1)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.shuffle_patch_size[0])

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        # assert self.input_size % self.mask_patch_size == 0
        # assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count + 1, dtype=int)
        mask[mask_idx + 1] = 1

        # mask_unrepeat = mask.reshape((self.rand_size, self.rand_size))
        # mask = mask_unrepeat.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask, mask


class CovGenerator:
    def __init__(self, input_size=192, mask_patch_size=32):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size

        assert self.input_size % self.mask_patch_size == 0
        # assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        # self.scale = self.mask_patch_size // self.model_patch_size

    def __call__(self, img):
        img_list = self.crop_image(img)
        size = int(len(img_list) ** 0.5)
        cov_list = [self.cal_covariance(patch).reshape(-1) for patch in img_list]
        cov_list = torch.stack([torch.stack(cov_list[i * size:(i + 1) * size]) for i in range(size)])
        cov_list = cov_list.permute(2, 0, 1).contiguous()
        # print(cov_list[:,7,:].flatten())
        return cov_list

    def cal_covariance(self, input):
        input = input / 255  # h w c
        input = input.transpose((2, 0, 1))
        h, w = input.shape[1], input.shape[2]
        # img = img.reshape((3, -1))
        input = input.reshape((3, -1))
        mean = input.mean(1)

        input = input - mean.reshape(3, 1)

        covariance_matrix = np.matmul(input, np.transpose(input))
        covariance_matrix = covariance_matrix / (h * w - 1)
        return torch.FloatTensor(covariance_matrix)

    def crop_image(self, image):
        # numpy will change the high width position when storing the image.
        image = np.array(image, np.float32)
        high, width = image.shape[0], image.shape[1]
        assert width % self.shuffle_patch_size[0] == 0 and high % self.shuffle_patch_size[1] == 0
        high_num, width_num = high // self.shuffle_patch_size[1], width // self.shuffle_patch_size[0]
        crop_x = [self.shuffle_patch_size[0] * i for i in range(wid_num + 1)]
        crop_y = [self.shuffle_patch_size[1] * i for i in range(high_num + 1)]
        im_list = np.array([])
        for i in range(len(crop_y) - 1):
            for j in range(len(crop_x) - 1):
                im_list.append(image[crop_y[i]:min(crop_y[i + 1], width), crop_x[j]:min(crop_x[j + 1], high), :])
        return im_list


class MyTransform:
    def __init__(self, config):

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        self.common_aug = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            # T.Resize((args.resize_size, args.resize_size),interpolation=str_to_pil_interp('bicubic')),
            # T.RandomCrop((args.img_size, args.img_size)),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.ColorJitter(0.3),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(),
        ])

        self.base = T.Compose([
            self.common_aug,
            self.to_tensor,
        ])

        self.mask_img = T.Compose([
            self.common_aug,
            self.to_tensor,
            T.RandomErasing(p=1, scale=(0.15, 0.45), ratio=(0.3, 3.3)),
        ])

        self.swap_img = T.Compose([
            self.common_aug,
            self.to_tensor,
            #T.RandomErasing(p=1, scale=(0.15, 0.45), ratio=(0.3, 3.3)),
            RandomSwap(config.TRAIN.NUM_PART)
        ])

        self.mask_swap_img = T.Compose([
            self.common_aug,
            self.to_tensor,
            T.RandomErasing(p=1, scale=(0.15, 0.45), ratio=(0.3, 3.3)),
            RandomSwap(config.TRAIN.NUM_PART)
        ])

        # self.swap_img = T.Compose([
        #     RandomSwap(config.TRAIN.NUM_PART)
        # ])

        self.mask = config.TRAIN.MASK
        self.swap = config.TRAIN.SWAP
        self.model = config.TRAIN.MODEL
        model_patch_size = 16

        # self.mask_generator = MaskGenerator(
        #     input_size=args.img_size,
        #     mask_patch_size=model_patch_size,
        #     model_patch_size=model_patch_size,
        #     mask_ratio=0.6,
        # )

        # self.cov_generator = CovGenerator(
        #     input_size=args.img_size,
        #     mask_patch_size=model_patch_size
        # )

    def __call__(self, image):
        imgs = []
        if self.model == 'base':
            imgs.append(self.base(image))
        elif self.model == 'mask_only':
            imgs.append(self.mask_img(image))
        elif self.model == 'swap_only':
            imgs.append(self.swap_img(image))
        elif self.model == 'full_m':
            # img = self.mask_img(image)
            # imgs.append(img)
            # imgs.append(self.swap_img(img))
            imgs.append(self.mask_img(image))
            imgs.append(self.swap_img(image))
        elif self.model == 'full_b':
            # img = self.mask_img(image)
            # imgs.append(img)
            # imgs.append(self.swap_img(img))
            imgs.append(self.base(image))
            imgs.append(self.mask_swap_img(image))
        # mask, mask_unrepeat = self.mask_generator()
        return imgs


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader(config, logger, is_train=True):
    if is_train:
        transform = MyTransform(config)
    else:
        size = int((256 / 224) * config.DATA.IMG_SIZE)
        transform = T.Compose([
            T.Resize(size, interpolation=str_to_pil_interp('bicubic')),
            # T.Resize((args.resize_size, args.resize_size),interpolation=str_to_pil_interp('bicubic')),
            T.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))])

    batch_size = config.DATA.BATCH_SIZE if is_train else config.DATA.EVAL_BATCH_SIZE
    logger.info(f'Pre-train data transform:\n{transform}')

    data_root = os.path.join(config.DATA.DATA_PATH, config.DATA.DATASET)
    if config.DATA.DATASET == 'imagenet':
        dataset = data_root
    elif config.DATA.DATASET == 'CUB':
        dataset = CUB(root=data_root, is_train=is_train, transform=transform)
    elif config.DATA.DATASET == 'dogs':
        dataset = dogs(root=data_root, train=is_train, transform=transform)
    elif config.DATA.DATASET == 'AFD':
        dataset = AFD(root=data_root,is_train=is_train,transform=transform)
    elif config.DATA.DATASET == 'air':
        dataset = FGVC_aircraft(root=data_root, is_train=is_train, transform=transform)
    else:
        dataset = Cultivar(root=data_root, is_train=is_train, transform=transform)

    if config.DATA.DATASET == "cifar10":
        num_classes = 10
    elif config.DATA.DATASET == "cifar100":
        num_classes = 100
    elif config.DATA.DATASET == "soyloc":
        num_classes = 200
    elif config.DATA.DATASET == "cotton":
        num_classes = 80
    elif config.DATA.DATASET == "dogs":
        num_classes = 120
    elif config.DATA.DATASET == "CUB":
        num_classes = 200
    elif config.DATA.DATASET == "car":
        num_classes = 196
    elif config.DATA.DATASET == 'air':
        num_classes = 100
    elif config.DATA.DATASET == "soybean2000":
        num_classes = 1938
    elif config.DATA.DATASET == "soybean_gene":
        num_classes = 1110
    elif config.DATA.DATASET == "AFD":
        num_classes = 4
    elif config.DATA.DATASET == "WRD":
        num_classes = 3
    elif config.DATA.DATASET[:13] == "soybean_aging":
        num_classes = 198
    elif config.DATA.DATASET == 'BTF':
        num_classes = 10
    else:
        raise NotImplementedError("Not in supported dataset list.")

    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=is_train)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=True, drop_last=is_train, collate_fn=collate_fn)

    return dataloader, num_classes


def build_loader_origin(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
