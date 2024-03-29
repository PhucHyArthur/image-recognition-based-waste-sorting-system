import os
import numpy as np
import json


def xml_path_to_image_path(xml_path):
    xml_path = xml_path.replace('\\\\', '/')
    xml_path = xml_path.replace('.xml', '.png')
    xml_parts = xml_path.split('/')

    xml_parts[-2] = ''
    xml_parts[-3] = ''
    xml_parts[-4] = 'images'

    return '/'.join([part for part in xml_parts if part != ''])


root_folder = 'D:/DNIIT/DoNaAI/Data_set'
label_folder = root_folder + '/annotations/CAM'
labels = os.listdir(label_folder)

train_paths = {}
validation_paths = {}
test_paths = {}

for label in labels:
    xml_path = label_folder + '/' + label
    xml_files = os.listdir(xml_path)

    if label not in train_paths:
        train_paths[label] = []
        validation_paths[label] = []
        test_paths[label] = []

    validation_size = int(0.2 * len(xml_files))
    test_size = int(0.2 * len(xml_files))
    random_nums = []
    while len(random_nums) != validation_size + test_size:
        num = np.random.randint(0, len(xml_files))
        if num not in random_nums:
            random_nums.append(num)
    for num in set(range(0, len(xml_files))) - set(random_nums):
        image_path = xml_path_to_image_path(xml_path + '/' + xml_files[num])
        train_paths[label].append(image_path)

    for _ in range(validation_size):
        num = random_nums.pop(0)
        image_path = xml_path_to_image_path(xml_path + '/' + xml_files[num])
        validation_paths[label].append(image_path)

    for _ in range(test_size):
        num = random_nums.pop(0)
        image_path = xml_path_to_image_path(xml_path + '/' + xml_files[num])
        test_paths[label].append(image_path)


label_folder = root_folder + '/annotations/NET'
labels = os.listdir(label_folder)

for label in labels:
    xml_path = label_folder + '/' + label
    xml_files = os.listdir(xml_path)

    if label not in train_paths:
        train_paths[label] = []
        validation_paths[label] = []
        test_paths[label] = []

    validation_size = int(0.2 * len(xml_files))
    test_size = int(0.2 * len(xml_files))
    random_nums = []
    while len(random_nums) != validation_size + test_size:
        num = np.random.randint(0, len(xml_files))
        if num not in random_nums:
            random_nums.append(num)
    for num in set(range(0, len(xml_files))) - set(random_nums):
        image_path = xml_path_to_image_path(xml_path + '/' + xml_files[num])
        train_paths[label].append(image_path)

    for _ in range(validation_size):
        num = random_nums.pop(0)
        image_path = xml_path_to_image_path(xml_path + '/' + xml_files[num])
        validation_paths[label].append(image_path)

    for _ in range(test_size):
        num = random_nums.pop(0)
        image_path = xml_path_to_image_path(xml_path + '/' + xml_files[num])
        test_paths[label].append(image_path)


data_paths = {
    'train': train_paths,
    'validation': validation_paths,
    'test': test_paths
}

with open('data_paths2.json', 'w', encoding='utf8') as fp:
    json.dump(data_paths, fp, ensure_ascii=False)