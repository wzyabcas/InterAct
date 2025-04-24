import os

import json
import shutil

data_root = './data'

datasets = ['omomo']

for dataset in datasets:
    dataset_path = os.path.join(data_root, dataset)


    sequence_folder = os.path.join(dataset_path, 'sequences')
    sequence_seg_folder = os.path.join(dataset_path, 'sequences_seg')
    omomo_text_folder = os.path.join(dataset_path, 'raw', 'omomo_text_anno_json_data')
    # read sequences
    json_names = os.listdir(omomo_text_folder)
    for json_name in json_names:
        # read json
        json_path = os.path.join(omomo_text_folder, json_name)
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        sequence_name = json_name.split('.')[0]
        sequence_path = os.path.join(sequence_folder, sequence_name)
        sequence_seg_path = os.path.join(sequence_seg_folder, sequence_name)
        text_path = os.path.join(sequence_seg_path, 'text.txt')
        human_path = os.path.join(sequence_path, 'human.npz')
        object_path = os.path.join(sequence_path, 'object.npz')
        if not os.path.exists(human_path) or not os.path.exists(object_path):
            continue
        os.makedirs(sequence_seg_path, exist_ok=True)
        # os.rename(human_path, os.path.join(sequence_seg_path, 'human.npz'))
        # os.rename(object_path, os.path.join(sequence_seg_path, 'object.npz'))  
        shutil.copy(human_path, os.path.join(sequence_seg_path, 'human.npz'))
        shutil.copy(object_path, os.path.join(sequence_seg_path, 'object.npz'))    
        # copy and seg text.txt
        with open(text_path, 'w') as f:
            f.write(json_content[sequence_name])
        print("Saved text to", text_path)
