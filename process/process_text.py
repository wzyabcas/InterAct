import os
import pandas as pd
import numpy as np
import re
import spacy
import codecs as cs

datasets = ['behave']

data_root = './data'

# Add ActionSet list
ActionSet = ['Rotate', 'Move', 'Carry', 'Hold', 'Play', 
             'Manipulate', 'Sit', 'Bowling', 'Lift', 'Eat', 
             'Adjust', 'Swing', 'Pass', 'Exercise', 'Kick', 'Drink']

nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

def remove_punctuation(text):
    text = text.replace('"', '')
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.replace('\'', '')
    text = text.replace('\r\n', '\r')
    text = text.replace('\n', '\r')
    text_segs = re.split(r'[\r;]', text.strip())
    return text_segs

def split_text(text_seg):
    text_seg_info = text_seg.split('#')
    text_content = text_seg_info[0].strip()
    text_start_frame = int(text_seg_info[1].replace('.', ''))
    text_end_frame = int(text_seg_info[2].replace('.', ''))

    return text_content, text_start_frame, text_end_frame

def remove_non_alphanumeric(text):
    pattern = re.compile('[^a-zA-Z0-9\s\.,!?;:"\'\(\)]')
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = cleaned_text.replace(u'\xa0', u' ')
    return cleaned_text

# Add function for action text processing
def process_action_text(text):
    pattern = re.compile('[^a-zA-Z]')
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

for dataset in datasets:
    dataset_path = os.path.join(data_root, dataset)
    # read texts.csv
    text_path_1 = os.path.join(data_root,'annotation/natural',f'{dataset}.csv')
    text_path_2 = os.path.join(data_root,'annotation/change',f'{dataset}.csv')
    text_path_3 = os.path.join(data_root,'annotation/shorten',f'{dataset}.csv')
    # Add path for action CSV
    action_path = os.path.join(data_root,'annotation/action',f'{dataset}.csv')
    
    sequence_folder = os.path.join(dataset_path, 'sequences')
    sequence_seg_folder = os.path.join(dataset_path, 'sequences_seg')
    os.makedirs(sequence_seg_folder, exist_ok=True)

    text_df_1 = pd.read_csv(text_path_1)
    texts_1 = text_df_1['Answer.action'].tolist()
    video_urls = text_df_1['Input.video_url'].tolist()

    text_df_2 = pd.read_csv(text_path_2)
    texts_2 = text_df_2['Answer.action'].tolist()
    
    text_df_3 = pd.read_csv(text_path_3)
    texts_3 = text_df_3['Answer.action'].tolist()
    
    # Read action data
    action_df = pd.read_csv(action_path)
    actions = action_df['Answer.action'].tolist()
    action_video_urls = action_df['Input.video_url'].tolist()
    
    # Create mapping from video URL to action
    action_map = {}
    for url, action in zip(action_video_urls, actions):
        action_map[url] = action

    # read sequences
    sequence_names = os.listdir(sequence_folder)
    for sequence_name in sequence_names:
        sequence_path = os.path.join(sequence_folder, sequence_name)
        human_path = os.path.join(sequence_path, 'human.npz')
        object_path = os.path.join(sequence_path, 'object.npz')
        if not os.path.exists(human_path) or not os.path.exists(object_path):
            continue
        human_motion = np.load(human_path, allow_pickle=True)
        object_motion = np.load(object_path, allow_pickle=True)
        for video_url, text_1, text_2, text_3 in zip(video_urls, texts_1, texts_2, texts_3):
            result_name = video_url.split('_')[0] + '_' + sequence_name + '_' + video_url.split('_')[-1]
            if result_name == video_url:
                try:
                    text_segs_1 = remove_punctuation(text_1)
                    text_segs_2 = remove_punctuation(text_2)
                    text_segs_3 = remove_punctuation(text_3)

                    for i, text_seg in enumerate(text_segs_1):
                        if text_seg == '':
                            continue
                        # text_seg: text#start_frame#end_frame
                        text_content, text_start_frame, text_end_frame = split_text(text_seg)
                        assert text_start_frame < text_end_frame
                        text_content_2 = text_segs_2[i].split('#')[0].strip()
                        text_content_3 = text_segs_3[i].split('#')[0].strip()
                        
                        text_content = remove_non_alphanumeric(text_content)
                        text_content_2 = remove_non_alphanumeric(text_content_2)
                        text_content_3 = remove_non_alphanumeric(text_content_3)


                        if dataset == 'chairs':
                            text_start_frame = text_start_frame*3
                            text_end_frame = text_end_frame*3


                        # create a new folder for the sequence_seg
                        sequence_seg_name = sequence_name + '_{}'.format(text_start_frame)
                        sequence_seg_path = os.path.join(sequence_seg_folder, sequence_seg_name)
                        os.makedirs(sequence_seg_path, exist_ok=True)
                        # copy and seg human.npz and object.npz
                        human_seg_path = os.path.join(sequence_seg_path, 'human.npz')
                        object_seg_path = os.path.join(sequence_seg_path, 'object.npz')
                        human_seg_motion = {}
                        object_seg_motion = {}
                        if len(human_motion['trans']) < text_start_frame:
                            print('Length error:', video_url)
                            continue
                        for key in human_motion.keys():
                            if key == 'betas' or key == 'vtemp' or key == 'gender':
                                human_seg_motion[key] = human_motion[key]
                            else:
                                human_seg_motion[key] = human_motion[key][text_start_frame:text_end_frame]
                        for key in object_motion.keys():
                            if key == 'name':
                                object_seg_motion[key] = object_motion[key]
                                obj_name = str(object_motion[key])  # Get object name for action text
                            else:
                                object_seg_motion[key] = object_motion[key][text_start_frame:text_end_frame]
                        np.savez(human_seg_path, **human_seg_motion)
                        np.savez(object_seg_path, **object_seg_motion)
                        
                        # Generate action.npy and action.txt if action exists for this video
                        if video_url in action_map:
                            action = process_action_text(action_map[video_url])
                            
                            # Create one-hot encoding for action
                            if action in ActionSet:
                                action_index = ActionSet.index(action)
                                action_onehot = np.zeros(len(ActionSet))
                                action_onehot[action_index] = 1
                                action_npy_path = os.path.join(sequence_seg_path, 'action.npy')
                                np.save(action_npy_path, action_onehot)
                            
                            # Create action.txt file
                            action_txt_path = os.path.join(sequence_seg_path, 'action.txt')
                            action_lower = action.lower()
                            if dataset.upper() == 'CHAIRS':
                                content = f'A person {action_lower} the chair.'
                            else:
                                content = f'A person {action_lower} the {obj_name.lower()}.'
                            with open(action_txt_path, 'w') as f:
                                f.write(content)
                        
                        # copy and seg text.txt
                        text_seg_path = os.path.join(sequence_seg_path, 'text.txt')

                        with cs.open(text_seg_path, 'w') as f:
                            content = ""
                            for text in [text_content, text_content_2, text_content_3]:
                                word_list, pose_list = process_text(text)
                                tokens = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
                                content += '%s#%s#%s#%s\r' % (text, tokens, 0.0, 0.0)
                            f.write(content.strip())
                except Exception as e:
                    print(e)
                    print('error:', video_url)
                    print('text_1:', text_1)
                    print('text_2:', text_2)
                    print('text_3:', text_3)
                    continue
                        

                

        

    
