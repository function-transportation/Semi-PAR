from model.solider.solider import make_model, parse_args
import yaml
from evaluate import real_data_evaluate
import glob
import torch

class UnbiasModelFinder:
    def __init__(self):
        self.args = parse_args()
        with open('./model/solider/config/configs/pa100k.yaml', 'r') as filehandle:
            self.cfg = yaml.load(filehandle,Loader=yaml.Loader)
            
            
    def find_unbias_model(self, target_paths):
        for model_path in target_paths:
            print(f'evaluating {model_path}')
            try:
                model = make_model(self.cfg, self.args, model_path)
                real_data_evaluate(eval_dirs=['../../DATA/person_image_00006',
                                            '../../DATA/person_image_00033',
                                            './data/DATA/person_image_select/C3_0819PM_HD',
                                            './data/DATA/person_image_select/C109_0819PM_HD',
                                            './data/DATA/person_image_select/A101_0819PM_HD',
                                            './data/DATA/person_image_select/O123_0819PM_HD'],
                                model=model,
                                attr_num=26)
            except RuntimeError:continue
            
if __name__ == '__main__':
    umf = UnbiasModelFinder()
    umf.find_unbias_model(glob.glob('/home/ubuntu/workspace_yamanishi/ALM-pedestrian-attribute/exp/pa100k/fixmatch/**/**.pth.tar'))
    exit()
    umf.find_unbias_model(glob.glob('../CameraX/pretrained/pa100k/*.pth.tar'))
    umf.find_unbias_model(f'./exp/pa100k/fixmatch/solider_maleplus_u075/{i}.pth.tar' for i in range(25, 65, 5))
    umf.find_unbias_model(f'./exp/pa100k/fixmatch/solider_maleplus_u07/{i}.pth.tar' for i in [25, 30, 35])    
    umf.find_unbias_model(f'./exp/pa100k/fixmatch/expo_0.6/{i}.pth.tar' for i in range(10, 35, 5))
        