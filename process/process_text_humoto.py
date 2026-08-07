import os
import yaml
import json
import shutil
import numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
from process.utils import ObjectSequence, ObjectsSequence, HumanSequence, Sequence, Processor
# dict_path = {
#     'MOTION_PATH': './data/humoto/sequences',
#     'OBJECT_PATH': './data/humoto/objects',
#     'MODEL_PATH': './models',
#     'MOTION_PATH_RAW_HUMAN': './data/humoto/raw/smplh',
#     'MOTION_PATH_RAW_OBJECT': './data/humoto/raw/output_process',
#     'OBJECT_PATH_RAW': './data/humoto/raw/humoto_objects_0805',
#     'PATH_RAW': './data/humoto/raw'
# }
dict_test_path = {
    'MOTION_PATH': './data/humoto_test/sequences',
    'OBJECT_PATH': './data/humoto_test/objects',
    'METADATA_PATH': './data/humoto_test/raw/humoto_0805',
    'MODEL_PATH': './models',
    'MOTION_PATH_RAW_HUMAN': './data/humoto_test/raw/smplh',
    'MOTION_PATH_RAW_OBJECT': './data/humoto_test/raw/output_process',
    'OBJECT_PATH_RAW': './data/humoto_test/raw/humoto_objects_0805',
    'PATH_RAW': './data/humoto_test/raw',
    'PATH_ROOT': './data/humoto_test/',
}
dict_path = dict_test_path  # Switch between test and full dataset by changing this line
procer = Processor(dict_path, smpl_init=False)  

datasets = ['humoto']
for dataset in datasets:

    # follow the naming convention of other process_text files
    sequence_folder = dict_path['MOTION_PATH']
    sequence_seg_folder = os.path.join(dict_path['PATH_ROOT'], 'sequences_seg')
    metadata_folder = dict_path['METADATA_PATH'] # metadata_folder / sequence_name / sequence_name.yaml is the metadata file for this->sequence.

    if os.path.isdir(sequence_seg_folder):
        raise FileExistsError(
            f"Directory {sequence_seg_folder} already exists.\nPlease delete it."
        )
    os.makedirs(sequence_seg_folder, exist_ok=False)
    logging.info(f"Created sequence_seg_folder directory: {sequence_seg_folder}")
    if not os.path.isdir(sequence_folder) or not os.path.isdir(metadata_folder):
        raise FileNotFoundError(f"Missing required folders in dataset {dataset}: 'sequences' or 'humoto_0805'.")

    sequence_names_at_the_beginning = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(dict_path['MOTION_PATH_RAW_HUMAN'])
        if filename.endswith('.npz')
    ]
    sequence_names_after_humoto_process = [
        path.name
        for path in Path(dict_path["MOTION_PATH"]).iterdir()
        if path.is_dir()
        and (path / "human.npz").is_file()
    ]
    if set(sequence_names_at_the_beginning) != set(sequence_names_after_humoto_process):
        raise ValueError(
            f"Mismatch between sequences in MOTION_PATH_RAW_HUMAN and MOTION_PATH: "
            f"{set(sequence_names_at_the_beginning)} vs {set(sequence_names_after_humoto_process)}"
        )
    sequence_names = sequence_names_at_the_beginning

    success = []; sequence_failures = []; segment_failures = []
    for name in sequence_names:
        try:
            yaml_path = os.path.join(
                metadata_folder,
                f"{name}",
                f"{name}.yaml"
            )
            with open(yaml_path, 'r') as f:
                seg_info = yaml.safe_load(f)
                start_frame = seg_info['start_frame']; end_frame = seg_info['end_frame']
                objects:List[str] = seg_info['objects']
                long_script:List[Dict[str, Union[str, int]]] = seg_info['long_script']
                short_script:str = seg_info['short_script']
                seg_times = len(long_script)


            objs_sequence: ObjectsSequence = ObjectsSequence(os.path.join(sequence_folder, name))
            human_sequence:HumanSequence = procer._load_human_sequence(sequence_folder, name, True)
            sequence = Sequence(
                start_frame=start_frame,
                end_frame=end_frame,
                human_sequence=human_sequence,
                objs_sequence=objs_sequence,
                text_description=long_script,
                extra_text_description=short_script,
                seg_times=seg_times
            )

        except Exception as exc:
            logging.exception(f"Failed to load sequence {name}: {exc}")
            sequence_failures.append((name, 'sequence', repr(exc)))
            print(f"[SUMMARY] {name}: 0 saved, sequence loading failed.")
            continue


        saved_count = 0; failed_count = 0
        for i in range(sequence.seg_times):

            seg_name = f"{name}_{sequence.text_description[i]['start_frame']}_{sequence.text_description[i]['end_frame']}_{i:03d}"
            try:
                sequence_seg_path = os.path.join(sequence_seg_folder, seg_name)
                text_path = os.path.join(sequence_seg_path, 'text.txt')
                human_path = os.path.join(sequence_seg_path, 'human.npz')
                os.makedirs(sequence_seg_path, exist_ok=True)

                this_seg_start_frame = sequence.text_description[i]['start_frame']
                this_seg_end_frame = sequence.text_description[i]['end_frame']
                this_seg_text = sequence.text_description[i]['script']

                # HUMOTO YAML is 1-based and includes the end frame, while a Python
                # slice is 0-based and excludes its end.
                slice_start = this_seg_start_frame - 1
                slice_end = this_seg_end_frame  # slice_end is exclusive, so we don't subtract 1
                human_sequence_seg:HumanSequence = sequence.human_sequence.slice(slice_start, slice_end)
                human_sequence_seg.dump(human_path)
                objs_sequence_seg:ObjectsSequence = sequence.objs_sequence.slice(slice_start, slice_end)
                objs_sequence_seg.dump(sequence_seg_path)

                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(this_seg_text)

            except Exception as exc:
                logging.exception(f"Failed to process segment {seg_name}: {exc}")
                segment_failures.append((seg_name, 'segment', repr(exc)))
                failed_count += 1
            else:
                success.append(seg_name)
                saved_count += 1
                print(f"[SAVED] {seg_name}")
        print(
            f"[SUMMARY] {name}: "
            f"{saved_count} saved, "
            f"{failed_count} failed."
        )


    print(
        f"[FINAL SUMMARY] "
        f"{len(success)} segments saved, "
        f"{len(sequence_failures)} sequences failed to load, "
        f"{len(segment_failures)} segments failed."
    )
    if sequence_failures or segment_failures:
        raise RuntimeError(
            "Humoto text processing completed with failures; "
            "see the log above."
        )


            
