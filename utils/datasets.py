import os
import sys
from PIL import Image, ImageOps
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.randaugment import RandAugmentMC
import cv2
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
import random
import math
import torchvision.transforms.functional as TF
from torchvision import transforms

class RandomResolutionChange(object):
    def __init__(self, min_resolution=2000*2000, max_resolution=8000*8000, p=0.5):
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # ランダムな解像度（総ピクセル数）を生成
            new_resolution = random.randint(self.min_resolution, self.max_resolution)
            
            # 元の画像のアスペクト比を取得
            w, h = img.size
            aspect_ratio = w / h
            
            # 新しい幅と高さを計算
            new_w = int(math.sqrt(new_resolution * aspect_ratio))
            new_h = int(new_resolution / new_w)
            
            # アスペクト比を維持しつつ、目標の解像度に最も近くなるように調整
            if abs(new_w * new_h - new_resolution) > abs((new_w-1) * (new_h+1) - new_resolution):
                new_w -= 1
                new_h += 1
            
            # リサイズ
            img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
            
            # 元のサイズに戻す
            img = TF.resize(img, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
        
        return img

def default_loader(path):
    return Image.open(path).convert('RGB')

class RandGaussianBlur(object):
    def __init__(self, p, radius=2):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if torch.rand(1) < self.p:
            kernel_size= np.random.randint(2, 6)*2-1
            sigma = np.random.uniform(0, 2)
            return transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        return img
    
class RandColorJitter(object):
    def __init__(self, p,):
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            brightness_factor = np.random.uniform(0.5,1.5)
            contrast_factor=np.random.uniform(0.5,1.5)
            saturation_factor = np.random.uniform(0.5,1.5)
            hue_factor = np.random.uniform(-0.1, 0.1)
            img = transforms.functional.adjust_brightness(img, brightness_factor)
            img = transforms.functional.adjust_contrast(img, contrast_factor)
            img = transforms.functional.adjust_saturation(img, saturation_factor)
            img = transforms.functional.adjust_hue(img, hue_factor)
        return img
    
class ResizeWithRatio:
    def __init__(self, size=(256, 128)):
        self.final_size = size

    def __call__(self, img):
        width, height = img.size
        final_height, final_width = self.final_size

        if height / width >= 2:
            # 縦が横の2倍以上の場合
            new_height = final_height
            new_width = int(width * final_height / height)
            img = img.resize((new_width, new_height), Image.BILINEAR)
            padding = (final_width - new_width) // 2
            color = img.getpixel((0, new_height//2))
            img = ImageOps.expand(img, (padding, 0, final_width - new_width - padding, 0), fill=color)
        else:
            # 縦が横の2倍以下の場合
            new_width = final_width
            new_height = int(height * final_width / width)
            img = img.resize((new_width, new_height), Image.BILINEAR)
            padding = (final_height - new_height) // 2
            color = img.getpixel((0, new_height//2))
            img = ImageOps.expand(img, (0, padding, 0, final_height - new_height - padding), fill=color)
        
        return img

class TransformFixMatch(object):
    def __init__(self, mean, std):
        crop_size=16
        self.weak = transforms.Compose([
            #ResizeWithRatio(size=(272, 136)),
            transforms.Resize(size=(288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(256, 128),
                                  padding=int(crop_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            #ResizeWithRatio(size=(272, 136)),
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
    
class SimpleDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        image_paths: 画像のパスのリスト
        transform: 画像に適用する変換（前処理）
        """
        self.image_paths = image_paths
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
        #transforms.Resize(size=(256, 128)),
        ResizeWithRatio(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image 
    
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
attr_nums['gender_age'] = 5
attr_nums['carry'] = 2
attr_nums['gender'] = 1 #2
attr_nums['poc2'] = 6

description = {}
# description['pa100k'] = ['Female',
#                         'AgeOver60',
#                         'Age18-60',
#                         'AgeLess18',
#                         'Front',
#                         'Side',
#                         'Back',
#                         'Hat',
#                         'Glasses',
#                         'HandBag',
#                         'ShoulderBag',
#                         'Backpack',
#                         'HoldObjectsInFront',
#                         'ShortSleeve',
#                         'LongSleeve',
#                         'UpperStride',
#                         'UpperLogo',
#                         'UpperPlaid',
#                         'UpperSplice',
#                         'LowerStripe',
#                         'LowerPattern',
#                         'LongCoat',
#                         'Trousers',
#                         'Shorts',
#                         'Skirt&Dress',
#                         'boots']

description['pa100k'] = ['Hat',
 'Glasses',
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
 'boots',
 'HandBag',
 'ShoulderBag',
 'Backpack',
 'HoldObjectsInFront',
 'AgeOver60',
 'Age18-60',
 'AgeLess18',
 'Female',
 'Front',
 'Side',
 'Back']

description['gender_age'] = ['Female',
                             'AgeLess18',
                             'AgeYoung',
                             'AgeOld',
                             'AgeOver60']

description['carry'] = ['Carry Bag',
                        'Baby Car']

description['age'] = ['AgeLess18',
                      'AgeYoung',
                      'AgeOld']

description['poc2'] = ['Female',
                        'Carry Bag',
                        'Baby Car',
                        'AgeLess18',
                        'AgeYoung',
                        'AgeOld']

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

description['gender'] = ['Female',]
                        #'condition_bad']

def peta_to_pa100k(attributes):
    peta_to_pa100k_d = {'AgeAbove61': 'AgeOver60', 
     'Hat': 'Hat', 
     'Sunglasses': 'Glasses', 
     'PlasticBags': 'HandBag', 
     'Messenger Bag': 'ShoulderBag', 
     'Backpack': 'Backpack',
     'Short Sleeve': 'ShortSleeve', 'Logo': 'UpperLogo', 'Plaid': 'UpperPlaid', 'Trousers': 'Trousers', 'Shorts': 'Shorts', 'Skirt': 'Skirt&Dress'}
    groups = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]
    attributes = np.array(attributes)
    sample_num, attr_numn = attributes.shape
    new_attributes = np.full((sample_num, 26), -1)
    new_attributes[:, 0] = 1-attributes[:, 16] #男女
    new_attributes[:, 2] = np.maximum.reduce([attributes[:, 1],attributes[:, 2],attributes[:, 3]])  #Age18-60
    new_attributes[:, 3] = (1-np.maximum.reduce([attributes[:, 1],attributes[:, 2],attributes[:, 3], attributes[:, 0]])) #AgeLess15
    new_attributes[:, 14] = (1-attributes[:, 26]) #Long Sleeve
    pa100k_attrs = [1,7, 8, 9,10, 11, 13, 16, 17, 22, 23, 24] 
    peta_attrs = [3, 10, 30,22, 17, 4, 26, 14, 21, 31, 25, 27]
    new_attributes = np.where(new_attributes>=2, -1, new_attributes)
    new_attributes[:, pa100k_attrs] = attributes[:, peta_attrs]
    new_attributes = new_attributes[:, groups]
    return new_attributes
    
class AgeDatasetSSL(data.Dataset):
    def __init__(self, root, label_file, transform = None, label=True, loader = default_loader, max_size=None, dataset='peta'):
        '''
        label_file: data_path, attr1, attr2, ...,attrN のスタイルのファイルを想定
        label: ラベルありの場合True, ラベルなしの場合False
        root: imageのroot
        '''
        with open(label_file, 'r') as f:
            files = f.readlines()
        self.dataset =dataset
        self.root = root
        image_paths = [f.split(',')[0] for f in files]
        if label:
            attributes =[]
            for f in files:
                attrs = f.split(',')[1:]
                print('attrs', attrs)
                attrs = [int(a) for a in attrs]#[f.split(',')[1:] for f in files]
                #print('attrs', attrs)
                attributes.append(attrs.index(1))
            self.attributes = attributes
            self.attributes_orig = attributes
            if self.dataset=='pa100k' and len(self.attributes[0])>30:
                self.attributes = peta_to_pa100k(attributes)
        self.image_paths = image_paths
        self.image_paths_orig = image_paths
        #print('orig_image_paths', self.image_paths)
        
        if max_size is not None and max_size!=-1:
            #print(max_size)
            indice = list(np.random.choice(np.arange(len(self.image_paths)), max_size))
            self.image_paths = [ip for i,ip in enumerate(self.image_paths) if i in indice]#list(np.array(self.image_paths)[indice])
            if label:
                self.attributes = list(np.array(self.attributes)[indice, :])
            
        self.root = root
        self.transform = transform
        self.loader = loader
        self.label = label
        
    def __getitem__(self, index):
        img_name=self.image_paths[index]
        img_name = img_name.replace('\n', '')
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        if self.label:
            label = self.attributes[index]
            return img, label
        else:
            return img, -1

    def __len__(self):
        return len(self.image_paths)
    
class MultiLabelDatasetSSL(data.Dataset):
    def __init__(self, root, label_file, transform = None, label=True, loader = default_loader, max_size=None, dataset='peta'):
        '''
        label_file: data_path, attr1, attr2, ...,attrN のスタイルのファイルを想定
        label: ラベルありの場合True, ラベルなしの場合False
        root: imageのroot
        '''
        
        with open(label_file, 'r') as f:
            files = f.readlines()
        self.dataset =dataset
        self.root = root
        image_paths = [f.split(',')[0] for f in files]
        if label:
            attributes =[]
            for f in files:
                attrs = f.split(',')[1:]
                attrs = [int(a) for a in attrs]#[f.split(',')[1:] for f in files]
                attributes.append(attrs)
            self.attributes = attributes
            self.attributes_orig = attributes
            if self.dataset=='pa100k' and len(self.attributes[0])>30:
                self.attributes = peta_to_pa100k(attributes)
        self.image_paths = image_paths
        self.image_paths_orig = image_paths
        #print('orig_image_paths', self.image_paths)
        
        if max_size is not None and max_size!=-1:
            #print(max_size)
            indice = list(np.random.choice(np.arange(len(self.image_paths)), max_size))
            self.image_paths = [ip for i,ip in enumerate(self.image_paths) if i in indice]#list(np.array(self.image_paths)[indice])
            if label:
                self.attributes = list(np.array(self.attributes)[indice, :])
            
        self.root = root
        self.transform = transform
        self.loader = loader
        self.label = label
        #print(self.image_paths)
        
    def select_image(self, target_attribute_ratios):
        total_image_count = len(self.image_paths_orig)
        selection_size = int(total_image_count* 1 / 2)
        prob = LpProblem("Image_Selection", LpMaximize)
        x = [LpVariable(f'x{i}', cat='Binary') for i in range(total_image_count)]
        prob += lpSum(x) == selection_size

        for ind, ratio in target_attribute_ratios.items():
            #print('ind ratio', ind, ratio)
            attribute_values = np.array([a[ind] for a in self.attributes_orig])
            num_zeros = sum(attribute_values == 0)
            num_ones = sum(attribute_values == 1)
            prob_ones = num_ones / (num_zeros+num_ones)
            random_labels = np.random.choice([0, 1], size=(attribute_values == -1).sum(), p=[1 - prob_ones, prob_ones])
            attribute_values[attribute_values == -1] = random_labels
            prob += lpSum([x[i] * attribute_values[i] for i in range(total_image_count)]) >= (ratio-0.02)*selection_size
            prob += lpSum([x[i] * attribute_values[i] for i in range(total_image_count)]) <= (ratio+0.02)*selection_size         
        prob.solve()
        selected_inds = [i for i in range(total_image_count) if value(x[i]) == 1]
        #print('selected_inds', selected_inds)
        self.image_paths = [self.image_paths_orig[i] for i in selected_inds]
        self.attributes = [self.attributes_orig[i] for i in selected_inds]
        #print(len(self.attributes))
        for i in range(len(self.attributes_orig[0])):
            print(i)
            print(sum([a[i] for a in self.attributes if a[i]!=-1]))
            print(len([a[i] for a in self.attributes if a[i]!=-1]))

    def __getitem__(self, index):
        img_name=self.image_paths[index]
        img_name = img_name.replace('\n', '')
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        if self.label:
            label = self.attributes[index]
            return img, torch.Tensor(label)
        else:
            if self.dataset=='peta':
                return img, torch.zeros(35)
            elif self.dataset=='pa100k':
                return img, torch.zeros(26)
            elif self.dataset == 'gender_age':
                return img, torch.zeros(5)
            elif self.dataset == 'carry':
                return img, torch.zeros(1)
            elif self.dataset == 'gender':
                return img, torch.zeros(1)

    def __len__(self):
        return len(self.image_paths)
    
def Get_age_Dataset(dataset, train_label_txt, train_unlabel_txt, test_label_txt, root, max_size=None, args=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        #ResizeWithRatio(size=(272, 136)),
        transforms.RandomHorizontalFlip(),
        RandGaussianBlur(p=0.5),
        RandColorJitter(p=0.5),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        #ResizeWithRatio(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])
    train_labeled_dataset = AgeDatasetSSL(
                                root=root, 
                                label_file=train_label_txt, 
                                transform=transform_labeled,
                                label=True,
                                max_size=args.label_size,
                                dataset=args.experiment
                            )
    train_unlabeled_dataset = AgeDatasetSSL(
                                root=root,
                                label_file=train_unlabel_txt,
                                transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                label=False,
                                max_size=args.unlabel_size,
                                dataset=args.experiment
                            )
    test_labeled_dataset = AgeDatasetSSL(
                                root=root,
                                label_file=test_label_txt,
                                transform=transform_val,
                                label=True,
                                max_size=max_size,
                                dataset=args.experiment
                            )
    return train_labeled_dataset, train_unlabeled_dataset, test_labeled_dataset, description[dataset]
 
def Get_SF_Dataset(dataset=None, train_label_txt=None, test_label_txt=None, root=None, max_size=None, args=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        RandomResolutionChange(min_resolution=3000, max_resolution=8000, p=0.3),
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        RandGaussianBlur(p=0.5),
        RandColorJitter(p=0.5),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    print('transform', transform_labeled)
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
        max_size=args.label_size if args.label_size != -1 else None,
        dataset=args.experiment
    )
    
    test_labeled_dataset = MultiLabelDatasetSSL(
        root=root,
        label_file=test_label_txt,
        transform=transform_val,
        label=True,
        max_size=max_size,
        dataset=args.experiment
    )
    
    return train_labeled_dataset, test_labeled_dataset, description[dataset]

def Get_fixmatch_Dataset(dataset, train_label_txt, train_unlabel_txt, test_label_txt, root, max_size=None, args=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        # ResizeWithRatio(size=(272, 136)),
        transforms.RandomHorizontalFlip(),
        RandGaussianBlur(p=0.5),
        RandColorJitter(p=0.5),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        # ResizeWithRatio(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])
    train_labeled_dataset = MultiLabelDatasetSSL(
                                root=root, 
                                label_file=train_label_txt, 
                                transform=transform_labeled,
                                label=True,
                                max_size=args.label_size,
                                dataset=args.experiment
                            )
    train_unlabeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=train_unlabel_txt,
                                transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                label=False,
                                max_size=args.unlabel_size,
                                dataset=args.experiment
                            )
    test_labeled_dataset = MultiLabelDatasetSSL(
                                root=root,
                                label_file=test_label_txt,
                                transform=transform_val,
                                label=True,
                                max_size=max_size,
                                dataset=args.experiment
                            )
    return train_labeled_dataset, train_unlabeled_dataset, test_labeled_dataset, description[dataset]

def Get_gaa_Dataset(dataset, train_label_txt, train_unlabel_txt, test_label_txt, root, max_size=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_labeled = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        # ResizeWithRatio(size=(272, 136)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        # ResizeWithRatio(size=(256, 128)),
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
        # ResizeWithRatio(size=(272, 136)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(256, 128),
                              padding=int(16*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        #ResizeWithRatio(size=(256, 128)),
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
        #ResizeWithRatio(size=(272, 136)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        #ResizeWithRatio(size=(256, 128)),
        transforms.ToTensor(),
        normalize
        ])
    train_dataset = MultiLabelDataset(root='data_path',
                label='train_list_path', transform=transform_train)
    val_dataset = MultiLabelDataset(root='data_path',
                label='val_list_path', transform=transform_test)
    return train_dataset, val_dataset, attr_nums[experiment], description[experiment]
    if experiment == 'pa100k':
        train_dataset = MultiLabelDataset(root='data_path',
                    label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                    label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['pa100k'], description['pa100k']
    elif experiment == 'gender_age':
        train_dataset = MultiLabelDataset(root='data_path',
            label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                    label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['gemder_age'], description['gender_age']
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
