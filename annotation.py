import os
import numpy as np
class AnnotationMaker:
    def __init__(self, label_file_dir='./data/label', image_dir='../../'):
        self.label_file_dir = label_file_dir
        self.image_dir = image_dir
        
    def update_annotation(self, label_file: str, target_directory: str):
        '''
        label_file(既にannotationがつけてある)にtarget_directoryのpathを追加する
        '''
        with open(label_file, 'a') as f:
            for root, dirs, files in os.walk(target_directory):
                #print(root, len(files))
                if len(files)>3:
                    rand_index = np.random.choice(len(files), size=3, replace=False)
                    for i in rand_index:
                        f.write(os.path.join(root, files[i]))
                        f.write('\n')
                        #print(os.path.join(root, files[i]))
                else:
                    for file in files:
                        f.write(os.path.join(root, file))
                        f.write('\n')
                        #print(os.path.join(root, f))
                
            #total_files += len(files)
    
if __name__=='__main__':
    annotation_maker = AnnotationMaker()
    annotation_maker.update_annotation('./data/label/clean_unlabel_annotation.txt', '../../DATA/person_image_select/')
