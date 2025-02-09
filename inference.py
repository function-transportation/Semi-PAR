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
import argparse
from tqdm import tqdm


# font_path = r"G:\Projects and Work\SAI\Person Attribute Retrieval\iccv19_attribute\font\arial.ttf"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--checkpoint_path', type=str, default=None)
args = parser.parse_args()

dataset='peta'
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
    pretrained_model=args.dataset
    model_path = args.checkpoint_path
print(model_path)
image_to_test = "/home/ubuntu/DATA/person_image_big2"
peta_age_indices = [0,1,2,3]
peta_gender_idx = 16
rap_age_indices = [1,2,3]
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


def default_loader(path):
    return Image.open(path).convert('RGB')


def par_results_pa100k(output):
    for i in range(len(output[0])):
        if i == 0:
            if output[0][i] == 1:
                print("Female")
            else:
                print("Male")
        else:
            if output[0][i] == 1:
                print(description["pa100k"][i])

def par_results_peta(output):
    outputs = []
    flag = True
    for i in range(len(output[0])):
        if i in peta_age_indices:
            if output[0][i] == 1:
                flag = False
                #print(description["peta"][i])
                outputs.append(description["peta"][i])
        elif i == peta_gender_idx:
            if output[0][i] == 0:
                outputs.append('Female')
                #print("Female")
            else:
                #print("Male")
                outputs.append('Male')
        else:
            if flag == True:
                #print("AgeLess15")
                outputs.append('AgeLess15')
                flag = False
            if (output[0][i] == 1):
                #print(description["peta"][i])
                outputs.append(description["peta"][i])
    return outputs

def par_results_rap(output):
    flag = True
    for i in range(len(output[0])):
        if i == 0:
            if output[0][i] == 1:
                print("Female")
            else:
                print("Male")
        elif i in rap_age_indices:
            if output[0][i] == 1:
                flag = False
                print(description["rap"][i])
        else:
            if flag == True:
                print("AgeOver46")
                flag = False
            if (output[0][i] == 1):
                print(description["rap"][i])


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])

model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums[pretrained_model])
model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = False
cudnn.deterministic = True

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

image_files = os.listdir(image_to_test)
image_files.sort()
start = time.time()
i=0

small_images = []
areas = []

for k in range(8):
    if k>1:continue
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
    batch_size=20
    for i in tqdm(range(0, len(small_images)//batch_size)):
        inputs = []
        for j in range(batch_size):
            image_path = small_images[i*batch_size+j]
            img = default_loader(image_path)
            img = transform_test(img)
            img = torch.unsqueeze(img, dim=0)
            input = img.cuda(non_blocking=True)
            inputs.append(input)
            
        inputs = torch.cat(inputs)
        #print(inputs.shape)
        output = model(inputs)


        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        for out in output:
            out = out.reshape(1, -1)
            #print(out.shape)
            if dataset=='pa100k':
                par_results_pa100k(out)
            elif dataset=='peta':
                outputs = par_results_peta(out)
            elif dataset=='rap':
                par_results_rap(out)
                print("\n")
            #print(outputs)
            if 'Male' in outputs:male_count+=1
            else:female_count+=1
            #print(male_count, female_count)
        #i+=1
        #if i==100:break
    print(k, male_count, female_count)
    end = time.time()
    print(end - start)