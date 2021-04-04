mnist(){
    python comparison_poc_2.py --dataset mnist --num-nodes 125 \
	   --labels  'FL,$\tau$=1 (full)' 'FL,$\tau$=50 (full)' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi $dpi \
	   --epochs 199 --histories \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_50.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_60_lut_50_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_70_lut_50_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_90_lut_50_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_100_lut_50_rounds_1.pkl \
	   --name comparison_clf_fcn_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_varying_lut_50_rounds_1.$format
}

fmnist(){
    python comparison_poc_2.py --dataset fmnist --num-nodes 125 \
	   --labels  'FL,$\tau$=1 (full)' 'FL,$\tau$=50 (full)' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi $dpi \
	   --epochs 199 --histories \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_50.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_60_lut_50_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_70_lut_50_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_90_lut_50_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_100_lut_50_rounds_1.pkl \
	   --name comparison_clf_fcn_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_varying_lut_50_rounds_1.$format
}

if [ $2 = 'jpg' ]; then
    dpi=100
    format='jpg'
else
    dpi=300
    format='eps'
fi


$1
