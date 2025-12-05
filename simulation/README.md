Place InterAct datasets as shown below:

```
hoi
├── data
│   ├── beahve
│   │   ├── objects
│   │       ├── backpack
│   │       ├── basketball  
            ...
│   │   └── sequences_canonical
│   │       ├── Date01_Sub01_chairblack_hand_1136
            ...
│   ├── imhd
│   │   ├── objects
│   │       ├── baseball
│   │       ├── broom  
            ...
│   │   └── sequences_canonical
│   │       ├── 20230827_zhaochf_bat_bat_twohands_swing4_0_0_2
│           ...
│   ...
├── models
...
```

To convert smpl motion sequences in a dataset to simulation, run

```bash
python smpl_to_simulation.py --dataset_name [dataset]
```

- The .pt fiels will be stored under `../InterAct/{dataste}`
- The .xml files will be stored under `../InterAct/{dataste}/{model_type}`
- The .urdf files will be stored under `../InterAct/{dataste}/objects`
