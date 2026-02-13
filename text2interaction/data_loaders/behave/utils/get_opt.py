import os
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.behave.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, use_global=False, wo_obj_motion=False):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.use_global = use_global
    opt.wo_obj_motion = wo_obj_motion

    if opt.dataset_name == 't2m_behave':
        opt.data_root =  '/projects/bcnt/ziyin/InterAct/data'#'../data' #'./data'
        opt.dim_pose = 553 + 6
        opt.markers_num = 77
        opt.max_motion_length = 300
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt