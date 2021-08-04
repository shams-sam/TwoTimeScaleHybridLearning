mnist(){
    python comparison_poc_2.py --dataset mnist --num-nodes 125 \
	   --labels  'FL,$\tau$=1 (full)' 'FL,$\tau$=50 (full)' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi 300 \
	   --epochs 199 --histories \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_50.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_60_lut_50_rounds_1_nocsi.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_70_lut_50_rounds_1_nocsi.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_90_lut_50_rounds_1_nocsi.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_100_lut_50_rounds_1_nocsi.pkl \
	   --name comparison_clf_svm_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_varying_lut_50_rounds_1_nocsi.png
}

fmnist(){
    python comparison_poc_2.py --dataset fmnist --num-nodes 125 \
	   --labels  'FL,$\tau$=1 (full)' 'FL,$\tau$=50 (full)' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi 300 \
	   --epochs 199 --histories \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_50.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_60_lut_50_rounds_1_nocsi.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_70_lut_50_rounds_1_nocsi.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_90_lut_50_rounds_1_nocsi.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_100_lut_50_rounds_1_nocsi.pkl \
	   --name comparison_clf_svm_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_varying_lut_50_rounds_1_nocsi.png
}

$1
