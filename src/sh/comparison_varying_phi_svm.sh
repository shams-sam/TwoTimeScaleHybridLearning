mnist(){
    python comparison.py --dataset mnist --num-nodes 125 \
	   --labels  'fl w/o eut' 'fl w/ eut' '$\phi$=0.001' '$\phi$=0.1' '$\phi$=1.0' '$\phi$=10.0' --ncols 3 --dpi 100 \
	   --epochs 50 --histories \
	   clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.001_factor_2_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_1.0_factor_2_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_10.0_factor_2_eut_range_10_20.pkl \
	   --name comparison_clf_svm_paradigm_hl_uniform_True_non_iid_10_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_varying_factor_2_eut_range_10_20.jpg

    python comparison.py --dataset mnist --num-nodes 125 \
	   --labels  'fl w/o eut' 'fl w/ eut' '$\phi$=0.001' '$\phi$=0.1' '$\phi$=1.0' '$\phi$=10.0' --ncols 3 --dpi 100 \
	   --epochs 50 --histories \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16.pkl \
	   clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.001_factor_2_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.1_factor_2_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_1.0_factor_2_eut_range_10_20.pkl \
	   clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_10.0_factor_2_eut_range_10_20.pkl \
       --name comparison_clf_svm_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_varying_factor_2_eut_range_10_20.jpg

}

$1
