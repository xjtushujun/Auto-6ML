# Run train_fix_l2ac.py on CIFAR-10 under gamma_l=100 and gramma_u=100
# python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100  --epoch 500 --val-iteration 500 --out result/cifar10@N_1500_r_100_100_fix_l2ac --dataset cifar10 --workers 0

# Run train_fix_l2ac.py on CIFAR-10 under gamma_l=150 and gramma_u=150
# python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150  --epoch 500 --val-iteration 500 --out result/cifar10@N_1500_r_150_150_fix_l2ac --dataset cifar10 --workers 0

# Run train_fix_l2ac.py on CIFAR-10 under gamma_l=100 and gramma_u=1
# python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1  --epoch 500 --val-iteration 500 --out result/cifar10@N_1500_r_100_1_fix_l2ac --dataset cifar10 --workers 0

# Run train_fix_l2ac.py on CIFAR-10 under gamma_l=100 and gramma_u=100 (reverse)
# python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --reverse True  --epoch 500 --val-iteration 500 --out result/cifar10@N_1500_r_100_reverse_fix_l2ac --dataset cifar10 --workers 0

# Run train_fix_l2ac.py on CIFAR-100 under gamma_l=10 and gramma_u=10
# python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 150 --imb_ratio_l 10 --imb_ratio_u 10  --epoch 500 --val-iteration 500 --out result/cifar100@N_150_r_10_10_fix_l2ac --dataset cifar100 --workers 0

# Run train_fix_l2ac.py on CIFAR-100 under gamma_l=20 and gramma_u=20
# python train_fix_l2ac.py --gpu 0 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20  --epoch 500 --val-iteration 500 --out result/cifar100@N_150_r_20_20_fix_l2ac --dataset cifar100 --workers 0

# Run train_fix_l2ac.py on STL-100 under gamma_l=10
# python train_fix_l2ac.py --gpu 0 --dataset stl10 --num_max 450 --imb_ratio_l 10 --epoch 500 --val-iteration 500 --out result/stl10@N_450_r_10_l2ac --workers 0

# Run train_fix_l2ac.py on STL-100 under gamma_l=20
# python train_fix_l2ac.py --gpu 0 --dataset stl10 --num_max 450 --imb_ratio_l 20  --epoch 500 --val-iteration 500 --out result/stl10@N_450_r_20_l2ac --workers 0
