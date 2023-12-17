import os
import sys
from PIL import Image
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.randaugment import RandAugmentMC
import cv2

def default_loader(path):
    return Image.open(path).convert('RGB')

class TransformFixMatch(object):
    def __init__(self, mean, std):
        crop_size=16
        self.weak = transforms.Compose([
            transforms.Resize(size=(288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(256, 128),
                                  padding=int(crop_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize(size=(288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(256, 128),
                                  padding=int(crop_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, loader = default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


attr_nums = {}
attr_nums['pa100k'] = 26
attr_nums['rap'] = 51
attr_nums['peta'] = 35

description = {}
description['pa100k'] = ['Female',
                        'AgeOver60',
                        'Age18-60',
                        'AgeLess18',
                        'Front',
                        'Side',
                        'Back',
                        'Hat',
                        'Glasses',
                        'HandBag',
                        'ShoulderBag',
                        'Backpack',
                        'HoldObjectsInFront',
                        'ShortSleeve',
                        'LongSleeve',
                        'UpperStride',
                        'UpperLogo',
                        'UpperPlaid',
                        'UpperSplice',
                        'LowerStripe',
                        'LowerPattern',
                        'LongCoat',
                        'Trousers',
                        'Shorts',
                        'Skirt&Dress',
                        'boots']

description['peta'] = ['Age16-30',
                        'Age31-45',
                        'Age46-60',
                        'AgeAbove61',
                        'Backpack',
                        'CarryingOther',
                        'Casual lower',
                        'Casual upper',
                        'Formal lower',
                        'Formal upper',
                        'Hat',
                        'Jacket',
                        'Jeans',
                        'Leather Shoes',
                        'Logo',
                        'Long hair',
                        'Male',
                        'Messenger Bag',
                        'Muffler',
                        'No accessory',
                        'No carrying',
                        'Plaid',
                        'PlasticBags',
                        'Sandals',
                        'Shoes',
                        'Shorts',
                        'Short Sleeve',
                        'Skirt',
                        'Sneaker',
                        'Stripes',
                        'Sunglasses',
                        'Trousers',
                        'Tshirt',
                        'UpperOther',
                        'V-Neck']

description['rap'] = ['Female',
                        'AgeLess16',
                        'Age17-30',
                        'Age31-45',
                        'BodyFat',
                        'BodyNormal',
                        'BodyThin',
                        'Customer',
                        'Clerk',
                        'BaldHead',
                        'LongHair',
                        'BlackHair',
                        'Hat',
                        'Glasses',
                        'Muffler',
                        'Shirt',
                        'Sweater',
                        'Vest',
                        'TShirt',
                        'Cotton',
                        'Jacket',
                        'Suit-Up',
                        'Tight',
                        'ShortSleeve',
                        'LongTrousers',
                        'Skirt',
                        'ShortSkirt',
                        'Dress',
                        'Jeans',
                        'TightTrousers',
                        'LeatherShoes',
                        'SportShoes',
                        'Boots',
                        'ClothShoes',
                        'CasualShoes',
                        'Backpack',
                        'SSBag',
                        'HandBag',
                        'Box',
                        'PlasticBag',
                        'PaperBag',
                        'HandTrunk',
                        'OtherAttchment',
                        'Calling',
                        'Talking',
                        'Gathering',
                        'Holding',
                        'Pusing',
                        'Pulling',
                        'CarryingbyArm',
                        'CarryingbyHand']


class MultiLabelDatasetSSL(data.Dataset):
    def __init__(self, root, label_file, transform = None, label=True, loader = default_loader, max_size=None):
        '''
        label_file: data_path, attr1, attr2, ...,attrN のスタイルのファイルを想定
        label: ラベルありの場合True, ラベルなしの場合False
        root: imageのroot
        '''
        with open(label_file, 'r') as f:
            files = f.readlines()

        self.root = root
        image_paths = [f.split(',')[0] for f in files]
        if label:
            attributes =[]
            for f in files:
                attrs = f.split(',')[1:]
                attrs = [int(a) for a in attrs]#[f.split(',')[1:] for f in files]
                attributes.append(attrs)
            self.attributes = attributes
        self.image_paths = image_paths
        
        if max_size is not None:
            self.image_paths = self.image_paths[:max_size]
            if label:
                self.attributes = self.attributes[:max_size]
            
        self.root = root
        self.transform = transform
        self.loader = loader
        self.label = label

    def __getitem__(self, index):
        img_name=self.image_paths[index]
        img_name = img_name.replace('\n', '')
        img = self.loader(os.path.join(self.root, img_name))
        #print('img', img.size, np.array(img))
        if self.transform is not None:
            img = self.transform(img)
        #mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        #std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        #if len(img)==2:
            #print('img_0', img[0].shape)
            #img_weak = img[0]*std + mean
            #img_strong = img[1]*std + mean
            #img_weak = transforms.ToPILImage()(img_weak)
            #img_strong = transforms.ToPILImage()(img_strong)
            #img_weak.save(f'./data/augmentation/aug_weak_{index}.png')
            #img_strong.save(f'./data/augmentation/aug_strong_{index}.png')
        #else:
            #print(img.shape)
            #img = img*std + mean
            #img_save = transforms.ToPILImage()(img)
            #img_save.save(f'./data/augmentation/aug_{index}.png')
        #Image.fromarray(img_save).save('./data/augmentation/aug.png')
            #print('img_save', img_save.size)
        #print('transformed', img.shape)
        if self.label:
            label = self.attributes[index]
            return img, torch.Tensor(label)
        else:
            return img, torch.zeros(35)

    def __len__(self):
        return len(self.image_paths)
    
def Get_fixmatch_Dataset(dataset, train_label_txt, train_unlabel_txt, test_label_txt, root, max_size=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])
    train_labeled_dataset = MultiLabelDatasetSSL(
                                root=root, 
                                label_file=train_label_txt, 
                                transform=transform_labeled,
                                label=True,
                                max_size=max_size
                            )
    train_unlabeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=train_unlabel_txt,
                                transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                label=False,
                                max_size=max_size
                            )
    test_labeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=test_label_txt,
                                transform=transform_val,
                                label=True,
                                max_size=max_size
                            )
    return train_labeled_dataset, train_unlabeled_dataset, test_labeled_dataset, description[dataset]

def Get_gaa_Dataset(dataset, train_label_txt, train_unlabel_txt, test_label_txt, root, max_size=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])
    train_labeled_dataset = MultiLabelDatasetSSL(
                                root=root, 
                                label_file=train_label_txt, 
                                transform=transform_labeled,
                                label=True,
                                max_size=max_size
                            )
    train_unlabeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=train_unlabel_txt,
                                transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                label=True,
                                max_size=max_size
                            )
    test_labeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=test_label_txt,
                                transform=transform_val,
                                label=True,
                                max_size=max_size
                            )
    return train_labeled_dataset, train_unlabeled_dataset, test_labeled_dataset, description[dataset]


def Get_sfda_Dataset(dataset, target_label_txt, test_label_txt, root, max_size=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])
    target_unlabeled_dataset = MultiLabelDatasetSSL(
                                root=root, 
                                label_file=target_label_txt, 
                                transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                label=False,
                                max_size=max_size
                            )
    test_labeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=test_label_txt,
                                transform=transform_val,
                                label=True,
                                max_size=max_size
                            )
    return target_unlabeled_dataset, test_labeled_dataset, description[dataset]

def Get_Dataset(experiment, approach):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
        ])

    if experiment == 'pa100k':
        train_dataset = MultiLabelDataset(root='data_path',
                    label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                    label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['pa100k'], description['pa100k']
    elif experiment == 'rap':
        train_dataset = MultiLabelDataset(root='data_path',
                    label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                    label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['rap'], description['rap']
    elif experiment == 'peta':
        train_dataset = MultiLabelDataset(root='./data/',
                    label='./data_list/peta/PETA_train_list.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root='./data/',
                    label='./data_list/peta/PETA_train_list.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['peta'], description['peta']
