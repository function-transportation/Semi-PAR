import os
import time

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw, ImageFont
from torch.backends import cudnn
import torchvision.transforms as transforms
import model as models
from main import Weighted_BCELoss
from utils.datasets import description
from models.solider.model_factory import build_backbone,build_classifier
from models.solider.base_block import FeatClassifier
from collections import OrderedDict
from tqdm import tqdm

import argparse
def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name.replace('module.module.', '')  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
# font_path = r"G:\Projects and Work\SAI\Person Attribute Retrieval\iccv19_attribute\font\arial.ttf"
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--dataset', type=str, default='peta')
args = parser.parse_args()
dataset=args.dataset
if args.checkpoint_path is None:
    dataset ='peta'
    if dataset == 'pa100k':
        pretrained_model = 'pa100k'
        model_path = 'pretrained/pa100k/pa100k_epoch_8.pth.tar'
    elif dataset=='peta':
        pretrained_model = 'peta'
        model_path = '/peta/peta_epoch_31.pth.tar'
    elif dataset=='rap':
        pretrained_model = 'rap'
        model_path = 'pretrained/rap/rap_epoch_9.pth.tar'
else:
    pretrained_model='peta'
    model_path = args.checkpoint_path
print(model_path)
image_to_test = "/home/ubuntu/DATA/person_image_big2"
pa100k_age_indices = [19,20,21]
pa100k_gender_idx = 22
pa100k_new_age_indices = [26, 27, 28, 29, 30]
peta_age_indices = [1,2,3]
peta_gender_idx = 0
peta_age_indices = [30,31,32,33]
peta_gender_idx = 34
rap_age_indices = [1,2,3]
rap_gender_idx = 0
attr_nums = {}
attr_nums['pa100k'] = 26
attr_nums['rap'] = 51
attr_nums['peta'] = 35

description = {}
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

description['pa100k_age'] = ['Hat',
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
 'Back',
 'AgeLess15',
 'Age15-30',
 'Age30-45',
 'Age45-60',
 'AgeOver60',
 'formal']
description['peta'] =['Hat',
 'Muffler',
 'No accessory',
 'Sunglasses',
 'Long hair',
 'Casual upper',
 'Formal upper',
 'Jacket',
 'Logo',
 'Plaid',
 'Short Sleeve',
 'Stripes',
 'Tshirt',
 'UpperOther',
 'V-Neck',
 'Casual lower',
 'Formal lower',
 'Jeans',
 'Shorts',
 'Skirt',
 'Trousers',
 'Leather Shoes',
 'Sandals',
 'Shoes',
 'Sneaker',
 'Backpack',
 'CarryingOther',
 'Messenger Bag',
 'No carrying',
 'PlasticBags',
 'Age16-30',
 'Age31-45',
 'Age46-60',
 'AgeAbove61',
 'Male']
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

def default_loader(path):
    return Image.open(path).convert('RGB')


def par_results_pa100k(output):
    result = {}
    for i in range(len(output[0])):
        if i in pa100k_age_indices:
            if output[0][i] == 1:
                result['age'] = description["pa100k"][i]
        elif i == pa100k_gender_idx:
            if output[0][i] == 1:
                result['gender'] = 'Female'
                #print("Female")
            else:
                result['gender'] = 'Male'
                #print("Male")
        else:
            if output[0][i] == 1:
                result[description["pa100k"][i]] = 'Yes'
                #print(description["pa100k"][i])
            else:
                result[description["pa100k"][i]] = 'No'
                
    return result

def par_results_peta(output):
    flag = True
    result = {}
    for i in range(len(output[0])):
        if i in peta_age_indices:
            if output[0][i] == 1:
                flag = False
                result['age'] = description["peta"][i]
                
                
        elif i == peta_gender_idx:
            if output[0][i] == 0:
                result['gender'] = 'Female'
                #print("Female")
            else:
                result['gender'] = 'Male'
                #print("Male")
        else:
            if flag == True:
                result['age'] = "AgeLess15"
                #print("AgeLess15")
                flag = False
            if (output[0][i] == 1):
                result[description["peta"][i]]='Yes'
                #print(description["peta"][i])
            else:
                result[description["peta"][i]]='No'
    return result

                

def par_results_rap(output):
    flag = True
    result = {}
    for i in range(len(output[0])):
        if i == 0:
            if output[0][i] == 1:
                #print("Female")
                result['gender'] = 'Female'
            else:
                #print("Male")
                result['gender'] = 'Male'
                
        elif i in rap_age_indices:
            if output[0][i] == 1:
                flag = False
                #print(description["rap"][i])
                result['age'] = description["rap"][i]
        else:
            if flag == True:
                result['age'] = 'AgeOver46'
                #print("AgeOver46")
                flag = False
            if (output[0][i] == 1):
                result[description["rap"][i]] = 'Yes'
                #print(description["rap"][i])
                
            else:
                result[description["rap"][i]] = 'No'
                
    return result

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])

import yaml
with open('./model/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
    cfg = yaml.load(filehandle,Loader=yaml.Loader)
            
print('cfg', cfg)
backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device='cuda')
print('build backbone complete')
classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
    nattr=attr_nums[args.dataset],
    c_in=c_output,
    bn=cfg["CLASSIFIER"]["BN"],
    pool=cfg["CLASSIFIER"]["POOLING"],
    scale =cfg["CLASSIFIER"]["SCALE"]
)
    
model = FeatClassifier(backbone, classifier)
#model = torch.nn.DataParallel(model)

checkpoint = torch.load(model_path, map_location='cuda')
model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
model.cuda()
model.eval()

image_files = os.listdir(image_to_test)
image_files.sort()
start = time.time()
i=0

small_images = []
areas = []

for k in range(8):
    small_images = []
    for curDir, dirs, files in os.walk("/home/ubuntu/DATA/person_image_big2"):
        for file in files:
            cur_file = os.path.join(curDir, file)
            img = Image.open(cur_file)
            h,w = img.size
            areas.append(h*w)
            if 2000*k<=h*w<2000*(k+1):small_images.append(cur_file)
            
    #for curDir, dirs, files in os.walk("/home/ubuntu/DATA/person_image_big"):
    #    for file in files:
    male_count=female_count=0
    batch_size=10
    for i in tqdm(range(0, len(small_images)//batch_size)):
        inputs = []
        for j in range(batch_size):
            image_path = small_images[i*batch_size+j]
            img = default_loader(image_path)
            img = transform_test(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.cuda()
            inputs.append(img)
            
        inputs = torch.cat(inputs)
        #print(inputs.shape)
        output = model(inputs)[0][0]

        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        for out in output:
            out = out.reshape(1, -1)
            #print(out.shape)
            if dataset=='pa100k':
                outputs = par_results_pa100k(out)
            elif dataset=='peta':
                outputs = par_results_peta(out)
            elif dataset=='rap':
                par_results_rap(out)
                print("\n")
            #print(outputs)
            if outputs['gender']=='Male':male_count+=1
            else:female_count+=1
            #print(male_count, female_count)
        #i+=1
        #if i==100:break
    print(k, male_count, female_count)
    end = time.time()
    print(end - start)