
from torchvision import datasets, transforms
import torch.utils.data as data
from .random_erasing import RandomErasing 

def train_baseline_aug():
        transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transform_train_list


def val_baseline_aug():
        transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transform_val_list

def train_pcb_baseline_aug():
        transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transform_train_list
        
def val_pcb_baseline_aug():
        transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transform_val_list

def build_augmentation_train(args, instance):
        if args.erasing_p != False and args.erasing_p>0:
                instance = instance +  [RandomErasing(probability = args.erasing_p, mean=[0.0, 0.0, 0.0])]
        if args.color_jitter:
                instance = instance +  [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)]
        
        return  transforms.Compose(instance)

def build_augmentation_val(args, instance):

        return  transforms.Compose(instance)              

