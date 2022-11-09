# CAtNIPP: Context-Aware Attention-based Network for Informative Path Planning
A context-aware neural framework for adaptive informative path planning (IPP) problem.

## Run
#### Training
1. Install requirements at the bottom.
2. Set appropriate parameters in `parameters.py`, including `NUM_META_AGENT`, `CUDA_DEVICE`, `BATCH_SIZE` (recommand 256 for every 8GB VRAM).
3. Name your run with `FOLDER_NAME`.
4. Run `python driver.py`

#### Evaluation
1. Set appropriate parameters in `/eval/test_parameters.py`, including `FOLDER_NAME`, `NUM_TEST`, `TRAJECTORY_SAMPLING`, `SAVE_IMG_GAP`, etc.
2. Run `/eval/test_driver.py`

## Files
* `parameters.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the local network.
* `worker.py` Interact with environment and collect episode experience.
* `attention_net.py` Define context-aware attention-based network.
* `env.py` Informative path planning environment.
* `gp_ipp.py` Gaussian Process and metrics calculation.
* `/eval` Test files for evaluation, similar to training.
* `/classes` Utilities for generating graph, ground truth, etc.
* `/model` Trained model.

### Demo of trajectory sampling variant CAtNIPP
![ts_demo](./result/ts_demo.gif)

### Requirements
```
python>=3.6
numpy>=1.17
ray>=1.15  % Ray should match python version
pytorch>=1.7
scipy
scikit-learn
matplotlib
imageio
shapely
```

### Cite
```
@InProceedings{cao2022catnipp,
  title = {Context-Aware Attention-based Network for Informative Path Planning},
  author = {Cao, Yuhong and Wang, Yizhuo and Vashisth, Apoorva and Fan, Haolin and Sartoretti, Guillaume},
  booktitle = {6th Annual Conference on Robot Learning},
  year = {2022}
}
```

### Authors
[Yuhong Cao](https://github.com/caoyuhong001)\
[Yizhuo Wang](https://github.com/wyzh98)\
[Apoorva Vashisth](https://github.com/AccGen99)\
[Haolin Fan](https://github.com/FHL1998)\
[Guillaume Sartoretti](https://github.com/gsartoretti)
