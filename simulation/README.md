## Dependencies

The dependencies of [the InterMimic Project](https://github.com/Sirui-Xu/InterMimic?tab=readme-ov-file#dependencies) are needed to run this script.

## Converting data

To convert smpl motion sequences in a dataset to simulation, run

```bash
python smpl_to_simulation.py --dataset_name [dataset]
```

## Outputs

- The .pt fiels will be stored under `InterAct/{dataste}`
- The .xml files will be stored under `InterAct/{dataste}/{model_type}`
- The .urdf files will be stored under `InterAct/{dataste}/objects`
