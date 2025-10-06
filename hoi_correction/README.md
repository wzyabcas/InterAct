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

2: Optimize hand pose

- Optimize hand pose on Behave dataset

  ```python
  python optimize_hand_behave.py --dataset behave
  ```

- Optimize hand pose on OMOMO dataset

  ```
  python optimize.py --dataset omomo
  ```

  