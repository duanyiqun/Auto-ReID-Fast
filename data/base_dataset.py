
from torchvision import datasets, transforms
import torch.utils.data as data

from PIL import Image
import os
import os.path
from .trans_aug import *
from .dist_sampler import DistributedSampler
import utils.distributed.misc as misc

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



class baseline_dataset():
    def __init__(self, args):
        self.distrubuted = args.distributed
        self.train_all = ''
        if args.train_all:
            self.train_all = '_all'
        self.args = args
        train_aug_list = eval(args.aug_for_train)()
        val_aug_list = eval(args.aug_for_val)()
        self.data_transforms = {}
        self.data_transforms['train'] = build_augmentation_train(args, train_aug_list)
        print("\n-------------------------------------------------")
        print("train augmentation list is {}".format(train_aug_list))
        print('--------------------------------------------------\n')
        self.data_transforms['val'] = build_augmentation_train(args, val_aug_list)
        print("\n-------------------------------------------------")
        print("val augmentation list is {}".format(val_aug_list))
        print('--------------------------------------------------\n')
        self.train_set = ImageFolder(os.path.join(args.data_dir, 'train' + self.train_all), transform= self.data_transforms['train'])
        self.val_set = ImageFolder(os.path.join(args.data_dir, 'val'), transform= self.data_transforms['val'])
        self.train_classnames = self.train_set.classes
        self.val_classnames = self.val_set.classes
        
    def get_loader(self):

        if self.distrubuted:

            train_sampler = DistributedSampler(len(self.train_set))
            valid_sampler = DistributedSampler(len(self.val_set))

            print("total size of train and val is {} and {}".format(len(self.train_set), len(self.val_set)))
            
            world_size = misc.get_world_size()
            rank = misc.get_rank()
            print("sub batch size in rank {} is {}".format(rank, self.args.batch_size //world_size))
            

            train_loader = data.DataLoader(self.train_set, 
                                            batch_size=self.args.batch_size //world_size,
                                            sampler= train_sampler,
                                            shuffle=False, 
                                            num_workers=self.args.num_workers, 
                                            pin_memory=self.args.pin_memory)

            val_loader = data.DataLoader(self.val_set, 
                                            batch_size=self.args.batch_size //world_size,
                                            sampler= valid_sampler,
                                            shuffle=False, 
                                            num_workers=self.args.num_workers, 
                                            pin_memory=self.args.pin_memory)
            
            print('Distributed sampler successfully loaded continue...')

        else:
        
            train_loader = data.DataLoader(self.train_set, 
                                            batch_size=self.args.batch_size,
                                            shuffle=self.args.shuffle, 
                                            num_workers=self.args.num_workers, 
                                            pin_memory=self.args.pin_memory)
            val_loader = data.DataLoader(self.val_set, 
                                            batch_size=self.args.batch_size,
                                            shuffle=False, 
                                            num_workers=self.args.num_workers, 
                                            pin_memory=self.args.pin_memory)


        return train_loader, val_loader


