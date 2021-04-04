mnist(){
    python comparison_poc_5.py --dataset mnist --num-nodes 125 \
	   --labels  'Centralized' 'FL,$\tau$=50' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi 300 \
	   --fracs 0.01 0.05 0.1 0.15 --accuracy 0.59 --histories \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.15.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.15.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.15.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.15.pkl \
	   --baselines \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_20.pkl \
	   --name relative_central_fl_cost_accuracy_dyn_non_iid_1_{}_bar_smaller.eps
}

fmnist(){
    python comparison_poc_5.py --dataset fmnist --num-nodes 125 \
	   --labels  'Centralized' 'FL,$\tau$=50' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi 300 \
	   --fracs 0.01 0.05 0.1 0.15 --accuracy 0.59 --histories \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.01.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.05.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.1.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.01_D_0.15.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.05_D_0.15.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.1_D_0.15.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.15_D_0.15.pkl \
	   --baselines \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_20.pkl \
	   --name relative_central_fl_cost_accuracy_dyn_non_iid_1_{}_bar_smaller.eps
}



$1
