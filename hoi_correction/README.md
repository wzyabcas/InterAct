Please follow the following steps for HOI optimization

1: Optimize full HOI sequence

- Optimize full HOI sequence on InterCap dataset

  ```python
  python optimize_fullbody_intercap.py --dataset intercap  
  ```
- Optimize full HOI sequence on Behave dataset

  ```python
  python optimize_fullbody.py --dataset behave  
  ```

2: Scan jitterings and correct wrist pose

- Scan sequences with jittering wrist poses on OMOMO dataset:

  ```bash
  python scan_diff.py --dataset omomo
  ```

  The detected sequences are listed in scan_results.scv.
- Correct wrist pose on detected sequences on OMOMO dataset:

  ```bash
  python correct_wrist.py --dataset omomo 
  ```

3: Optimize hand pose

- Optimize hand pose on Behave dataset

  ```python
  python optimize_hand_behave.py --dataset behave
  ```
- Optimize hand pose on OMOMO dataset

  ```
  python optimize.py --dataset omomo
  ```
