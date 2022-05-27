import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil
# Remove empty file
def remove_empty_file():
    for i in range(1,3):
        root = f'./Sources/dataset{i}/obj_train_data/'
        path_file = [name for name in os.listdir(root) if name.endswith('.txt')]
        count = 0
        for file in path_file:
            filesize = os.path.getsize(root + file)
            if filesize == 0:
                id = file.split('.')[0]
                os.remove(root + file)
                os.remove(root + f'{id}.jpg')
                count += 1
        print(f'Remove empty file in folder Annotations{i}: ', count)

# remove_empty_file()

# Rename file
def rename_file():
    for i in range(1,3):
        root = f'./Sources/dataset{i}/obj_train_data/'
        path_file = [name for name in os.listdir(root) if name.endswith('.txt')]
        if i == 1:
            n = len(path_file)
        for id, file in tqdm(enumerate(path_file)):
            id_image = file.split('.')[0]
            if i == 2:
                id = n + id
            os.rename(root + file, root + f'{id}.txt')
            os.rename(f'./Sources/dataset{i}/obj_train_data/{id_image}.jpg', f'./Sources/dataset{i}/obj_train_data/{id}.jpg')

# Prepare data for yolo
def data_yolo():
    root = './Sources/final_data/obj_train_data/'
    path_file = [name for name in os.listdir(root) if name.endswith('jpg')]
    n_sample = len(path_file)

    b1, b2 = int(0.9 * n_sample), int(0.95 * n_sample)
    print(b1, b2)
    
    for name in path_file[:b1]:
        id = name.split('.')[0]
        shutil.copy(root + name, f'./Sources/final_data/yolo_data/train/images/{name}')
        shutil.copy(root + f'{id}.txt', f'./Sources/final_data/yolo_data/train/labels/{id}.txt')
        
    for name in path_file[b1:b2]:
        id = name.split('.')[0]
        shutil.copy(root + name, f'./Sources/final_data/yolo_data/test/images/{name}')
        shutil.copy(root + f'{id}.txt', f'./Sources/final_data/yolo_data/test/labels/{id}.txt')

    for name in path_file[b2:]:
        id = name.split('.')[0]
        shutil.copy(root + name, f'./Sources/final_data/yolo_data/valid/images/{name}')
        shutil.copy(root + f'{id}.txt', f'./Sources/final_data/yolo_data/valid/labels/{id}.txt')

    print('Done!')

data_yolo()