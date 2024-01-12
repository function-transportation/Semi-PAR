import torch
import pickle
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from models.solider.base_block import FeatClassifier
import yaml
import glob
from models.solider.model_factory import build_backbone,build_classifier
from collections import OrderedDict

image_to_test = "test"
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
attr_nums['pa100k_age'] = 32


description = {}


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

def par_results_pa100k_age(output):
    result = {}
    for i in range(len(output[0])):
        if i in pa100k_age_indices:
            continue
        elif i in pa100k_new_age_indices:
            if output[0][i] == 1:
                result['age'] = description["pa100k_age"][i]
        elif i == pa100k_gender_idx:
            if output[0][i] == 1:
                result['gender'] = 'Female'
                #print("Female")
            else:
                result['gender'] = 'Male'
                #print("Male")
        elif i==31:
            if output[0][i] == 0:
                result['style'] = 'casual'
                #print("Female")
            else:
                result['style'] = 'formal'
        else:
            if output[0][i] == 1:
                result[description["pa100k_age"][i]] = 'Yes'
                #print(description["pa100k"][i])
            else:
                result[description["pa100k_age"][i]] = 'No'
               
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

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v.to('cpu')
    return new_state_dict


class SOLIDER:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        transform, target_transform = get_transform()
        self.transform = transform
        self.target_transform = target_transform
        self.attr_num = attr_nums[dataset.lower()]
    

        if dataset =='pa100k':
            with open('./models/solider/config/configs/pa100k.yaml', 'r') as filehandle:
                cfg = yaml.load(filehandle,Loader=yaml.Loader)
        elif dataset=='peta':
            with open('./models/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
                cfg = yaml.load(filehandle,Loader=yaml.Loader)
                
        elif dataset=='pa100k_age':
            with open('./models/solider/config/configs/pa100k_age.yaml', 'r') as filehandle:
                cfg = yaml.load(filehandle,Loader=yaml.Loader)
                
        print('cfg', cfg)
        backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
        print('build backbone complete')
        classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
            nattr=self.attr_num,
            c_in=c_output,
            bn=cfg["CLASSIFIER"]["BN"],
            pool=cfg["CLASSIFIER"]["POOLING"],
            scale =cfg["CLASSIFIER"]["SCALE"]
        )
        
        model = FeatClassifier(backbone, classifier)
        
        if dataset=='peta':
            if args.device==torch.device('cpu'):
                model.load_state_dict(torch.load('pretrained/peta/ckpt_max_solider_small.pth', map_location='cpu')['state_dicts'],)
            elif args.device==torch.device('cuda'):
                model.load_state_dict(torch.load('pretrained/peta/ckpt_max_solider_small.pth', map_location='cuda')['state_dicts'],)
        elif dataset=='pa100k':
            #model = torch.nn.DataParallel(model)
            print(args.device)
            if args.device==torch.device('cpu'):
                model.load_state_dict(torch.load('pretrained/pa100k/ckpt_max_solider3.pth', map_location='cpu')['state_dicts'])
            elif args.device==torch.device('cuda'):
                #model = torch.nn.DataParallel(model)
                print('solider pa100k')
                model.load_state_dict(torch.load('pretrained/pa100k/ckpt_max_solider3.pth', map_location='cuda')['state_dicts'])
                #model = torch.nn.DataParallel(model)
                #model.load_state_dict(torch.load('pretrained/pa100k/ckpt_max_solider.pth', map_location='cuda')['state_dicts'],)
        elif dataset=='pa100k_age':
            print(args.device)
            if args.device==torch.device('cpu'):
                model.load_state_dict(torch.load('pretrained/pa100k_age/ckpt_max_2023-10-02_16:26:32.pth', map_location='cpu')['state_dicts'],)
            elif args.device==torch.device('cuda'):
                print('solider pa100k_age')
                solider_path = glob.glob('../SOLIDER-PersonAttributeRecognition/exp_result/PA100k_age/swin_s.sm08/img_model/*.pth')[-1]
                #model.load_state_dict(torch.load('pretrained/pa100k_age/ckpt_max_2023-10-02_16:26:32.pth', map_location='cuda')['state_dicts'],)
                model.load_state_dict(torch.load(solider_path, map_location='cuda')['state_dicts'],)
        
        model.to(args.device)
        model.eval()
        self.model = model
          
    @torch.no_grad()  
    def predict(self, img:Image.Image, decode=True):
        img = self.target_transform(img)
        img = img.unsqueeze(dim=0)
        if self.args.device==torch.device('cuda'):
            img = img.cuda()

        out = self.model(img)
        out = F.sigmoid(out[0][0])
        #print('out', out[0][-10:])
        threshold = torch.ones_like(out)/2
        if self.dataset == 'pa100k_age':
            threshold[0][-1] = 0.05
            threshold[0][pa100k_gender_idx] = 0.6
        #pred = torch.gt(out, torch.ones_like(out)/2) 
        pred = torch.gt(out, threshold)
        if decode:
            if self.dataset == 'pa100k':
                result = par_results_pa100k(pred)
            elif self.dataset == 'peta':
                result = par_results_peta(pred)
            elif self.dataset == 'rap':
                result = par_results_rap(pred)
            elif self.dataset == 'pa100k_age':
                result = par_results_pa100k_age(pred)
                
            return result     
        else:
            return pred
        
    def test_transform(self, img):
        return self.target_transform(img)
    
    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor, decode=True):
        #batch.cuda()
        out = self.model(batch)
        out = F.sigmoid(out[0][0])
        pred = torch.gt(out, torch.ones_like(out)/2) 
        results = []
        if decode:
            for p in pred:
                p = p.unsqueeze(0)
                if self.dataset == 'pa100k':
                    result = par_results_pa100k(p)
                elif self.dataset == 'peta':
                    result = par_results_peta(p)
                elif self.dataset == 'rap':
                    result = par_results_rap(p)
                elif self.dataset == 'pa100k_age':
                    result = par_results_pa100k_age(p)
                    
                assert "result" in locals()
                results.append(result)
            return results     
        else:
            return pred
        
         
def get_pkl_rootpath(dataset):
    root = os.path.join("./data", f"{dataset}")
    data_path = os.path.join(root, 'dataset.pkl')

    return data_path
   
def get_transform():
    #height = args.height
    #width = args.width
    height, width = 256, 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    

    return train_transform, valid_transform

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = T.Compose([
    T.Resize(size=(256, 128)),
    T.ToTensor(),
    normalize
])
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

if __name__ == '__main__':
    
    predictor = SOLIDER(dataset='peta')
    img = Image.open('/home/yamanishi/function/DATA/person_image/059/001/0.jpg')
    #attribute = predictor.predict(img)
    #print('attribute', attribute)
    
    imgs = [Image.open('/home/yamanishi/function/DATA/person_image/059/001/0.jpg') for _ in range(5000)]
    imgs = [transform_test(img) for img in imgs]
    
    for i in tqdm(range(len(imgs)//100)):
        batch = torch.stack(imgs[100*i:100*(i+1)],)
        results = predictor.predict_batch(batch)
    # for _ in tqdm(range(10000)):
    #     attribute = predictor.predict(img)
        
    # imgs = [Image.open('/home/yamanishi/function/DATA/person_image/059/001/0.jpg') for _ in range(5000)]
    # dataset = Dataset(imgs, transform_test)
    # dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    # for data in tqdm(dataloader):
    #     results = predictor.predict_batch(data)