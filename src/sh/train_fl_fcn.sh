python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
# python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.03 --non-iid 1 --repeat 1 --dry-run 0 &
# python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.06 --non-iid 1 --repeat 1 --dry-run 0 &

python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
# python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.03 --non-iid 10 --repeat 1 --dry-run 0 &
# python train_model.py --dataset mnist --clf fcn --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.06 --non-iid 10 --repeat 1 --dry-run 0 &


# Logging:  ../ckpts/mnist_125/logs/clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.03_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.03_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.06_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.06_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0.log
