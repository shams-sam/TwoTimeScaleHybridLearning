mnist(){
    python comparison_poc_3.py --dataset mnist --num-nodes 125 \
	   --labels  'Centralized' 'FL,$\tau$=50' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi 100 \
	   --fracs 0.06 0.12 0.25 0.5 --accuracy 0.8 --histories \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.06_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.12_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.25_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.5_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.06_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.12_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.25_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.5_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.06_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.12_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.25_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.5_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.06_D_0.5.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.12_D_0.5.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.25_D_0.5.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_T1_15_Tmax_40_E_0.5_D_0.5.pkl \
	   --baselines \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   --name cost_vs_accuracy_{}.png
}

mnist_dyn(){
    python comparison_poc_3.py --dataset mnist --num-nodes 125 \
	   --labels  'Centralized' 'FL,$\tau$=50' '$\Gamma$=1,$\tau$=60' '$\Gamma$=1,$\tau$=70' '$\Gamma$=1,$\tau$=90' '$\Gamma$=1,$\tau$=100' --ncols 3 --dpi 300 \
	   --fracs 0.06 0.12 0.25 0.5 --accuracy 0.75 --histories \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.12_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.25_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.5_D_0.06.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.12_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.25_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.5_D_0.12.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.12_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.25_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.5_D_0.25.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.5.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.12_D_0.5.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.25_D_0.5.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.5_D_0.5.pkl \
	   --baselines \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   --name cost_vs_accuracy_dyn_non_iid_1_{}.eps
}


$1
