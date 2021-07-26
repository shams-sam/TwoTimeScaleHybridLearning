#!/bin/bash

mnist(){	
    python comparison_poc_1.py --dataset mnist --num-nodes 125 \
	   --labels  'FL,$\tau$=1 (full)' 'FL,$\tau$=20 (full)' '$\Gamma$=1,$\tau$=20' '$\Gamma$=5,$\tau$=20' '$\Gamma$=10,$\tau$=20' '$\Gamma$=25,$\tau$=20' --ncols 3 --dpi 300 \
	   --epochs 100 --histories \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25_nocsi.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25_nocsi.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25_nocsi.pkl \
       --name comparison_clf_fcn_non_iid_varying_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_varying_nocsi.png
}

fmnist(){
    python comparison_poc_1.py --dataset fmnist --num-nodes 125 \
	   --labels  'FL,$\tau$=1 (full)' 'FL,$\tau$=20 (full)' '$\Gamma$=1,$\tau$=20' '$\Gamma$=5,$\tau$=20' '$\Gamma$=10,$\tau$=20' '$\Gamma$=25,$\tau$=20' --ncols 3 --dpi $dpi \
	   --epochs 100 --histories \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_3_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16.pkl \
	   clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_range_20.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_1.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_5.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_10.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_25.pkl \
       --name comparison_clf_fcn_non_iid_varying_num_workers_125_lr_0.01_decay_1e-05_batch_16_eut_20_lut_5_rounds_varying.$format
}

$1
