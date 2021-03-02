mnist(){
    python comparison.py --dataset mnist --num-nodes 125 \
	   --labels  'fl iid w/o eut' 'fl iid w/ eut' 'fl non-iid w/o eut' 'fl non-iid w eut' --ncols 2 --dpi 100 \
	   --epochs 50 --histories \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_10_20.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_10_20.pkl \
	   --name comparison_clf_fcn_paradigm_fl_uniform_True_non_iid_varying_num_workers_125_lr_0.01_decay_1e-05_batch_16.jpg

    python comparison.py --dataset mnist --num-nodes 125 \
	   --labels  'fl iid w/o eut' 'fl iid w/ eut' 'fl non-iid w/o eut' 'fl non-iid w eut' --ncols 2 --dpi 100 \
	   --epochs 50 --histories \
	   clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_10_20.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_10_20.pkl \
	   --name comparison_clf_svm_paradigm_fl_uniform_True_non_iid_varying_num_workers_125_lr_0.001_decay_1e-05_batch_16.jpg
}


$1
