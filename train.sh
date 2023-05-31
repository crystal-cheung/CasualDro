CUDA_VISIBLE_DEVICES=5 python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.1 --cons_train 0.999_0.9 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 5000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type dro --dataset logit_z --penalty_anneal_iters 5000


CUDA_VISIBLE_DEVICES=5 python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.2 --cons_train 0.999_0.9 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 5000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type dro --dataset logit_z --penalty_anneal_iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.1 --cons_train 0.999_0.8 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 5000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type dro --dataset logit_z --penalty_anneal_iters 5000


CUDA_VISIBLE_DEVICES=5 python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.2 --cons_train 0.999_0.8 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 5000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type dro --dataset logit_z --penalty_anneal_iters 5000

CUDA_VISIBLE_DEVICES=5 python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.1 --cons_train 0.999_0.7 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 5000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type dro --dataset logit_z --penalty_anneal_iters 5000


CUDA_VISIBLE_DEVICES=5 python main.py --l2_regularizer_weight 0.001 --lr 0.005 --noise_ratio 0.2 --cons_train 0.999_0.7 --cons_test 0.999_0.8_0.2_0.001 --penalty_weight 10000 --steps 5000 --dim_inv 5 --dim_sp 5 --data_num_train 5000 --data_num_test 5000 --n_restarts 1 --irm_type dro --dataset logit_z --penalty_anneal_iters 5000
