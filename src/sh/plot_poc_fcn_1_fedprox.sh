mnist(){
    python comparison_poc_1.py --dataset mnist --num-nodes 125 \
	   --labels  'FedProx,$\tau$=1 (full)' 'FedProx,$\tau$=20 (full)' '$\Gamma$=1,$\tau$=20' '$\Gamma$=5,$\tau$=20' '$\Gamma$=10,$\tau$=20' '$\Gamma$=25,$\tau$=20' --ncols 3 --dpi 300 \
	   --epochs 100 --histories \
	   clf_fcn_paradigm_fpn_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fpn_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25_nocsi.pkl \
	   clf_fcn_paradigm_fpn_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fpn_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25_nocsi.pkl \
	   clf_fcn_paradigm_fpn_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fpn_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10_nocsi.pkl \
	   clf_fcn_paradigm_fp_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25_nocsi.pkl \
       --name comparison_clf_fcn_non_iid_varying_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_varying_nocsi_for_fedprox.png
}

$1
