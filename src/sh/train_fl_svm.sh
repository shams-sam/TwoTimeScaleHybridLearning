# python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 1 --repeat 1 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.001 --non-iid 1 --repeat 1 --dry-run 0 &
# python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.006 --non-iid 1 --repeat 1 --dry-run 0 &

# Logging:  ../ckpts/mnist_125/logs/clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0.log

python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 100 --lr 0.01 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.001 --non-iid 10 --repeat 1 --dry-run 0 &
python train_model.py --dataset mnist --clf svm --paradigm fl --num-workers 125 --num-clusters 25 5 1 --epochs 25 --lr 0.006 --non-iid 10 --repeat 1 --dry-run 0 &

# Logging:  ../ckpts/mnist_125/logs/clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.006_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_0.log
# Logging:  ../ckpts/mnist_125/logs/clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0.log
