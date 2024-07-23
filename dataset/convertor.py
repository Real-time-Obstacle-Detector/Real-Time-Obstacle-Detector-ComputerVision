import os

base_dir = 'dataset'

#directories
os.makedirs(f'{base_dir}/images/train', exist_ok=True)
os.makedirs(f'{base_dir}/images/val', exist_ok=True)
os.makedirs(f'{base_dir}/images/test', exist_ok=True)
os.makedirs(f'{base_dir}/labels/train', exist_ok=True)
os.makedirs(f'{base_dir}/labels/val', exist_ok=True)
os.makedirs(f'{base_dir}/labels/test', exist_ok=True)

#now we need to Load annotations and classes
with open(f'{base_dir}/labels/_annotations.txt', 'r') as annotation_file:
    annotations = annotation_file.readlines()

'''this function will ensure that our annotation's information are in YoloV8 structured way
'''
def process_annotations(annotations, subset):
    for annotation in annotations:
        parts = annotation.strip().split()
        image_file = parts[0]
        bbox_data = parts[1:]

        label_file = os.path.splitext(image_file)[0] + '.txt'
        print(f'new annotation file: {base_dir}/labels/{subset}/{label_file}')
        with open(f'{base_dir}/labels/{subset}/{label_file}', 'w') as f:
            for bbox in bbox_data:
                f.write(f'{bbox.replace(",", " ")}\n')

process_annotations(annotations, 'train')
