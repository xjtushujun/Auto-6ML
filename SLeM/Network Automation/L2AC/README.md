# Imbalanced Semi-supervised Learning with Bias Adaptive Classifier

This repository contains code for the paper
**"Imbalanced Semi-supervised Learning with Bias Adaptive Classifier"** 
by Renzhen Wang, Xixi Jia, Quanziang Wang, Yichen Wu and Deyu Meng.

## Dependencies

* `python3`
* `pytorch == 1.10.0`
* `torchvision`
* `scipy`

## Scripts
Please check out `run.sh` for all the scripts to run our method (L2AC).

### Training procedure of L2AC
if you want to run train_fix_l2ac.py on CIFAR-10 with the same imbalance ratios (e.g., 100) between labeled and unlabeled data. 
```
python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100  --epoch 500 
--val-iteration 500 --out result/cifar10@N_1500_r_100_100_fix_l2ac --dataset cifar10 --workers 0
```

## Credit
1. https://github.com/bbuing9/DARP
2. https://github.com/ildoonet/pytorch-randaugment

## Citation
If you find our work useful for your research, please cite with the following bibtex:
```bibtex
@inproceedings{wangimbalanced,
  title={Imbalanced Semi-supervised Learning with Bias Adaptive Classifier},
  author={Wang, Renzhen and Jia, Xixi and Wang, Quanziang and Wu, Yichen and Meng, Deyu},
  booktitle={International Conference on Learning Representations}
  year = {2023},
}
```

## Questions
Please feel free to contact "rzwang@xjtu.edu.cn".
