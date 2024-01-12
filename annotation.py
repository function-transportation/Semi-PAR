import os
import numpy as np
import pickle
import pulp

class AnnotationMaker:
    def __init__(self, label_file_dir='./data/label', image_dir='../../'):
        self.label_file_dir = label_file_dir
        self.image_dir = image_dir
        self.labels = ['Age16-30','Age31-45','Age46-60','AgeAbove61','Backpack',
        'CarryingOther', 'Casual lower','Casual upper','Formal lower', 'Formal upper',
        'Hat','Jacket','Jeans','Leather Shoes','Logo',
        'Long hair','Male','Messenger Bag','Muffler','No accessory',
        'No carrying','Plaid','PlasticBags','Sandals','Shoes',
        'Shorts','Short Sleeve','Skirt','Sneaker','Stripes',
        'Sunglasses','Trousers','Tshirt','UpperOther','V-Neck']
        
    def update_annotation(self, label_file: str, target_directory: str):
        '''
        label_file(既にannotationがつけてある)にtarget_directoryのpathを追加する
        '''
        s = set()
        with open(label_file, 'a') as f:
            for root, dirs, files in os.walk(target_directory):
                for cam in ['A101', 'C3', 'C109', 'O123']:
                    if cam in root:continue
                #if 'PM' in root and 'O123' not in root:continue
                #elif 'AM' not in root and 'PM' not in root:continue
                #print(root.split('/')[-2], len(files))
                #s.add(root.split('/')[-2])
                if len(files)>3:
                    rand_index = np.random.choice(len(files), size=3, replace=False)
                    for i in rand_index:
                        f.write(os.path.join(root, files[i]))
                        #f.write(os.path.join(root, files[i]).replace('./data', '../..'))
                        f.write('\n')
                        #print(os.path.join(root, files[i]))
                else:
                    for file in files:
                        f.write(os.path.join(root, file))
                        #f.write(os.path.join(root, file).replace('./data', '../..'))
                        f.write('\n')
            #total_files += len(files)
            
    def add_label(self, label_file, pseudo_label_file):
        with open(pseudo_label_file, 'rb') as f:
            pseudo_labels =  pickle.load(f)
            
        with open(label_file, 'r') as f:
            original_lines = f.readlines()
        
        new_lines = []
        for i in range(len(original_lines)):
            tmp = original_lines[i].split(',')
            if len(tmp)==1:
                pseudo_label = pseudo_labels[i]
                tmp[0] = tmp[0].replace('\n', '')
                for label in pseudo_label:
                    tmp.append(str(label))
                tmp[-1]+='\n'
                #print(tmp)
            new_lines.append(','.join(tmp))
        #print(new_lines)
        #print(len(new_lines))
        
        with open(label_file, 'w') as f:
            for line in new_lines:
                f.write(line)
                
    def select_annotation(self, label_file,new_label_file,new_label_not_selected_file, new_sample_num=10000):
        initial_labels = []
        image_paths = []
        with open(label_file) as f:
            attributes = f.readlines()
            for attr in attributes:
                att = attr.replace('¥n', '').split(',')[1:]
                att = [int(float(a)) for a in att]
                image_path = attr.replace('¥n', '').split(',')[0]
                initial_labels.append(att)
                image_paths.append(image_path)
                
        A = np.array(initial_labels)
        num_sample, num_attribute = A.shape
        problem = pulp.LpProblem("index_selection", pulp.LpMaximize)
        W = [pulp.LpVariable(f"W_{i}", cat=pulp.LpBinary) for i in range(num_sample)]
        epsilon = 0.01
        
        target_attribute_ratio = {16: 0.5} #男女半々
        for j in [16]:
            problem += pulp.lpSum([W[i] * A[i][j] for i in range(num_sample)]) <= (target_attribute_ratio[j]+epsilon)*new_sample_num
            problem += pulp.lpSum([W[i] * A[i][j] for i in range(num_sample)]) >= (target_attribute_ratio[j]-epsilon)*new_sample_num

        problem += pulp.lpSum(W) == new_sample_num
        problem.solve(pulp.PULP_CBC_CMD(msg = False))
        W_optimal = np.array([pulp.value(var) for var in W]).astype(int)
        
        with open(label_file, 'r') as f:
            original_lines = f.readlines()
        print(sum(W_optimal))
        with open(new_label_file, 'w') as f:
            for i, line in enumerate(original_lines):
                if W_optimal[i]:
                    f.write(line)
                    
        with open(new_label_not_selected_file, 'w') as f:
            for i, line in enumerate(original_lines):
                if not W_optimal[i]:
                    f.write(line)
                    
    def split_annotation(self, old_label_file, new_label_file1, new_label_file2, split_ratio=0.8):
        initial_labels, ids, image_paths,  = [], [], []
        with open(old_label_file) as f:
            attributes = f.readlines()
            for attr in attributes:
                att = attr.replace('¥n', '').split(',')[1:]
                att = [int(float(a)) for a in att]
                image_path = attr.replace('¥n', '').split(',')[0]
                initial_labels.append(att)
                image_paths.append(image_path)
                ids.append(image_path.split('/')[4]+ '_'+ image_path.split('/')[5])
                
        split1_index = np.random.choice(list(set(ids)), int(len(set(ids))*split_ratio), replace=False)
        with open(new_label_file1, 'w') as f:
            for i, (line,id_) in enumerate(zip(attributes, ids)):
                if id_ in split1_index:
                    f.write(line)
                    
        with open(new_label_file2, 'w') as f:
            for i, (line,id_) in enumerate(zip(attributes, ids)):
                if id_ not in split1_index:
                    f.write(line)
                    
    def move_annotation(self, old_label_file, new_label_file, choose_num=500):
        initial_labels = []
        image_paths = []
        with open(old_label_file) as f:
            attributes = f.readlines()
            for attr in attributes:
                att = attr.replace('¥n', '').split(',')[1:]
                att = [int(float(a)) for a in att]
                image_path = attr.replace('¥n', '').split(',')[0]
                initial_labels.append(att)
                image_paths.append(image_path)
                
        male_index = [i for i in range(len(image_paths)) if initial_labels[i][16]==1]
        female_index = [i for i in range(len(image_paths)) if initial_labels[i][16]==0]
        print(len(male_index))
        print(len(female_index))
        choosen_male_index = np.random.choice(male_index, choose_num//2, replace=False)
        choosen_female_index = np.random.choice(female_index, choose_num//2, replace=False)
        choosen_index = np.concatenate([choosen_female_index, choosen_male_index])
        print(len(choosen_female_index))
        print(len(choosen_male_index))
        print(len(choosen_index))
        print(choosen_index.max())
        print(len(image_paths))
        count=0
        with open(new_label_file, 'w') as f:
            for i, line in enumerate(attributes):
                if i in choosen_index:
                    count+=1
                    f.write(line)
        print(count)
        count2=0
        with open(old_label_file, 'w') as f:
            for i, line in enumerate(attributes):
                if i not in choosen_index:
                    count2+=1
                    f.write(line)
        print(count2)
            
            
            
    
if __name__=='__main__':
    annotation_maker = AnnotationMaker()
    
    annotation_maker.add_label('./data/label/expo_annotation.txt', './data/pseudo_label_expo_pa100k.pkl')
    # annotation_maker.split_annotation('./data/label/clean_train_annotation_orig.txt', './data/label/clean_train_annotation.txt',
    #                                   './data/label/clean_eval_annotation.txt')
    #annotation_maker.update_annotation('./data/label/expo_annotation.txt', './data/DATA/person_image_select')
    #annotation_maker.add_label('./data/label/clean_unlabel_annotation_orig.txt')
    # annotation_maker.select_annotation(label_file='./data/label/clean_train_annotation.txt',
    #                                  new_label_file='./data/label/clean_train_annotation_solider.txt',
    #                                  new_label_not_selected_file='./data/label/clean_train_annotation_sub_solider.txt',
    #                                  new_sample_num=650)
    # annotation_maker.select_annotation(label_file='./data/label/clean_unlabel_annotation_orig.txt',
    #                                   new_label_file='./data/label/clean_unlabel_annotation_alm.txt', 
    #                                   new_label_not_selected_file='./data/label/clean_unlabel_annotation_sub_alm.txt',
    #                                   new_sample_num=30000)