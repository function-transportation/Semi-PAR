import os
from PIL import Image
import matplotlib.pyplot as plt

def get_image_areas(directory):
    areas = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    with Image.open(os.path.join(root, filename)) as img:
                        width, height = img.size
                        areas.append(width * height)  # 面積を計算
                except IOError:
                    pass  # ファイルが開けないか画像でない場合は無視
    return areas

def plot_histogram(areas1, areas2, areas3, areas4):
    plt.figure(figsize=(8, 6))
    alpha=0.3
    print(len(areas1), len(areas2), len(areas3), len(areas4))
    plt.hist(areas1, bins=30, color='blue', alpha=alpha)
    plt.hist(areas2, bins=30, color='red', alpha=alpha)
    plt.hist(areas3, bins=30, color='green', alpha=alpha)
    plt.hist(areas4, bins=30, color='orange', alpha=alpha)
    plt.title('Image Areas')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    plt.savefig('histgram.png')
    #plt.show()

# 画像が保存されているディレクトリを指定
directory = '/home/ubuntu/DATA/person_image_1920x1080'
areas1 = get_image_areas('/home/ubuntu/DATA/person_image_1920x1080')
areas2 = get_image_areas('/home/ubuntu/DATA/person_image_768x432')
areas3 = get_image_areas('/home/ubuntu/DATA/person_image_384x216')
areas4 = get_image_areas('/home/ubuntu/DATA/person_image_192x108')
plot_histogram(areas1, areas2, areas3, areas4)