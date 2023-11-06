import yaml
import glob
import torch
import pickle
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from easydict import EasyDict
import numpy as np
from model.inception_iccv import inception_iccv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

image_to_test = "test"
peta_age_indices = [0,1,2,3]
peta_gender_idx = 16
rap_age_indices = [1,2,3]
pa100k_age_indices = [1,2,3]
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
    result = {}
    for i in range(len(output[0])):
        if i in pa100k_age_indices:
            if output[0][i] == 1:
                result['age'] = description["peta"][i]
        elif i == 0:
            if output[0][i] == 1:
                result['gender'] = 'Female'
            else:
                result['gender'] = 'Male'
        else:
            if output[0][i] == 1:
                result[description["pa100k"][i]] = 'Yes'
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


normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = T.Compose([
    T.Resize(size=(256, 128)),
    T.ToTensor(),
    normalize
])

class ALM:
    def __init__(self, args, dataset):
        self.args = args
        self.device = args.device
        self.dataset = dataset

        model = inception_iccv(pretrained=True, num_classes=attr_nums[dataset], )
        model = torch.nn.DataParallel(model)
        #cudnn.benchmark = False
        #cudnn.deterministic = True
        if dataset=='pa100k':
            model_path = 'pretrained/pa100k/pa100k_epoch_8.pth.tar'
        elif dataset=='peta':
            model_path = 'pretrained/peta/peta_epoch_31.pth.tar'
        elif dataset=='rap':
            model_path = 'pretrained/rap/rap_epoch_9.pth.tar'
            
        self.test_transform = transform_test
        checkpoint = torch.load(model_path)
        
        model.load_state_dict(checkpoint['state_dict'])
        if args.device==torch.device('cpu'):
            model_cpu = inception_iccv(pretrained=True, num_classes=attr_nums[dataset],)
            model_cpu.load_state_dict(model.module.state_dict())
            model = model_cpu.cpu()
        else:
            model_cpu = inception_iccv(pretrained=True, num_classes=attr_nums[dataset],)
            model_cpu.load_state_dict(model.module.state_dict())
            model = model_cpu.cuda()
            #model = model.cuda()
        model.eval()
        self.model = model
        
    def predict(self, img: Image.Image):
        #img = img.convert('RGB')
        img = transform_test(img)
        img = torch.unsqueeze(img, dim=0)
        if self.device==torch.device('cpu'):
            input = img
        else:
            input = img.cuda(non_blocking=True)
        output = self.model(input)
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
        
        output = torch.sigmoid(output.data).cpu()
        return output
        output = np.where(output > 0.5, 1, 0)
        if self.dataset == 'rap':
            attribute = par_results_rap(output)
        elif self.dataset == 'pa100k':
            attribute = par_results_pa100k(output)
        elif self.dataset == 'peta':
            attribute = par_results_peta(output)
            
        return attribute
    
    def predict_batch(self, batch: torch.Tensor):
        batch =  batch.cuda(non_blocking=True)
        output = self.model(batch)
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
        out = F.sigmoid(output)
        return out.cpu()
        pred = torch.gt(out, torch.ones_like(out)/2) 
        results = []
        for p in pred:
            p = p.unsqueeze(0)
            if self.dataset == 'pa100k':
                result = par_results_pa100k(p)
            elif self.dataset == 'peta':
                result = par_results_peta(p)
            elif self.dataset == 'rap':
                result = par_results_rap(p)
            results.append(result)
        return results 
def calc_metric(pred, gt): 
    TP = torch.sum((pred == 1) & (gt == 1)).item()
    FP = torch.sum((pred == 1) & (gt == 0)).item()
    FN = torch.sum((pred == 0) & (gt == 1)).item()
    TN = torch.sum((pred == 0) & (gt == 0)).item()

    # Recall（再現率）の計算
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

    # Precision（適合率）の計算
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0

    # F1スコアの計算
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    # Accuracy（正答率）の計算
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0.0

    # 結果の表示
    #print(f"Recall: {recall}")
    #print(f"Precision: {precision}")
    #print(f"F1 Score: {f1_score}")
    #print(f"Accuracy: {accuracy}")  
    return recall, precision, f1_score, accuracy 
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = transform_test(img)
        
        return img
    
if __name__=='__main__':
    images = []
    labels = open('/home/yamanishi/function/ALM-pedestrian-attribute/data_list/peta/PETA_test_list.txt').readlines()
    label = []
    for line in labels:
        items = line.split()
        img_name = items.pop(0)
        if os.path.isfile(os.path.join('./data', img_name)):
            cur_label = torch.tensor([int(v) for v in items])
            label.append(cur_label)
            images.append(os.path.join('./data', img_name))
          
    label = torch.stack(label)
    
    images = [Image.open(im) for im in images]
    dataset = Dataset(images, transform_test)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    args = EasyDict()
    args.device=torch.device('cuda')
    model = ALM(args, dataset='peta')
    all_out = []
    attr_name = description['peta']
    print(attr_name)
    all_out = []
    with torch.no_grad():
        for i,data in tqdm(enumerate(dataloader)):
            output = model.predict_batch(data)
            all_out.append(output)
        
    all_out = torch.cat(all_out)
    with open('../DATA/par_result/solider_peta.pkl', 'wb') as f:
        pickle.dump(all_out, f) 
    print(all_out.shape)
    threshold = 0
    left = 0.1
    right = 0.9
    recall_dict = {}
    prec_dict = {}
    f1_dict = {}
    acc_dict = {}
    bests = []
    for i in range(attr_nums['peta']):
        score_all = []
        recalls, precisions, f1s, accuracies = [], [], [], []
        start = 0.1
        end = 0.9
        for thresh in torch.arange(start, end, 0.01):
            attr = all_out[:, i]
            pred = (attr>thresh).to(int)
            #print('pred', pred)
            gt = label[:, i]
            recall, precision, f1_score, accuracy  = calc_metric(pred, gt, )
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1_score)
            accuracies.append(accuracy)
            score_all.append((round(f1_score, 3), round(thresh.item(), 3)))
        #print(score_all)
        bests.append((i, description['peta'][i], sorted(score_all, reverse=True)[0],))
        print(i, sorted(score_all, reverse=True)[0])
        best_thresh = sorted(score_all, reverse=True)[0][1]
        fig, ax = plt.subplots()
        ax.plot(list(torch.arange(start, end, 0.01)), precisions, label='precisions')
        ax.plot(list(torch.arange(start, end, 0.01)), recalls, label='recall')
        ax.plot(list(torch.arange(start, end, 0.01)), f1s, label='f1_score')
        ax.plot(list(torch.arange(start, end, 0.01)),accuracies, label='accuracy')
        ax.vlines(best_thresh, 0, 1)
        ax.set_xticks(list(torch.arange(0.1, 1, 0.1)))
        ax.legend()
        
        attr_name = description['peta'][i]
        fig.savefig(f'./data/result/{attr_name}.jpg')
        recall_dict[attr_name] = recalls
        prec_dict[attr_name] = precisions
        f1_dict[attr_name] = f1s
        acc_dict[attr_name] = accuracies
        
    for b in bests:
        print(b)
        
with open('./data/result/recall_dict', 'wb') as f:
    pickle.dump(recall_dict, f)
    
with open('./data/result/prec_dict', 'wb') as f:
    pickle.dump(prec_dict, f)
    
with open('./data/result/f1_dict', 'wb') as f:
    pickle.dump(f1_dict, f)
    
with open('./data/result/acc_dict', 'wb') as f:
    pickle.dump(acc_dict, f)
        
        
                


            