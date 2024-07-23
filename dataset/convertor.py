import os
import random
import shutil
from PIL import Image

base_dir = 'dataset'

os.makedirs(f'{base_dir}/images/train', exist_ok=True)
os.makedirs(f'{base_dir}/images/val', exist_ok=True)
#os.makedirs(f'{base_dir}/images/test', exist_ok=True)
os.makedirs(f'{base_dir}/labels/train', exist_ok=True)
os.makedirs(f'{base_dir}/labels/val', exist_ok=True)
#os.makedirs(f'{base_dir}/labels/test', exist_ok=True)

with open(f'{base_dir}/labels/_annotations.txt', 'r') as annotation_file:
    annotations = annotation_file.readlines()

# copy a subset of images from train to val and test
def copy_files(file_list, src_dir, dest_dir):
    for file in file_list:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

# get a list of all image files
image_files = [f for f in os.listdir(f'{base_dir}/images/images') if f.endswith(('.jpg', '.jpeg', '.png'))]

# shuffle the list to ensure random distribution
random.shuffle(image_files)

# split into training, validation, and test sets
num_images = len(image_files)
num_val = num_images // 10
num_train = num_images // 90

val_files = image_files[:num_val]
train_filers = image_files[num_val:]

# copy files to respective directories
copy_files(val_files, f'{base_dir}/images/images', f'{base_dir}/images/val')
copy_files(train_filers, f'{base_dir}/images/images', f'{base_dir}/images/train')

# process annotations
def process_annotations(annotations, subset):
    for annotation in annotations:
        parts = annotation.strip().split()
        image_file = parts[0]
        bbox_data = parts[1:]

        # Skip if image_file not in subset directory
        if not os.path.exists(os.path.join(base_dir, f'images/{subset}', image_file)):
            continue

        label_file = os.path.splitext(image_file)[0] + '.txt'
        print(f'New annotation file: {base_dir}/labels/{subset}/{label_file}')
        
        # Get image dimensions
        image_path = os.path.join(base_dir, 'images', subset, image_file)
        image = Image.open(image_path)
        img_width, img_height = image.size

        with open(f'{base_dir}/labels/{subset}/{label_file}', 'w') as f:
            for bbox in bbox_data:
                data = bbox.split(",")
                
                #lets normalize coordinates
                x_center = float(data[0]) / img_width
                y_center = float(data[1]) / img_height
                width = float(data[2]) / img_width
                height = float(data[3]) / img_height
                
                # write the normalized coordinates with the class label
                f.write(f'{data[5]} {x_center} {y_center} {width} {height}\n')

process_annotations(annotations, 'train')
process_annotations(annotations, 'val')
#process_annotations(annotations, 'test')