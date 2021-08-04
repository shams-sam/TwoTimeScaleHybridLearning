mnist(){
    python comparison_poc_6.py --dataset mnist --num-nodes 125 \
	   --c1 5e-5 7.5e-5 1e-4 2.5e-4 5e-4 7.5e-4 1e-3 2.5e-3 5e-3 7.5e-3 1e-2 \
	   --c2 5e0 7.5e0 1e1 2.5e1 5e1 7.5e1 1e2 2.5e2 5e2 7.5e2 1e3 \
	   --c3 7.5e2 1e3 2.5e3 5e3 7.5e3 1e4 2.5e4 5e4 7.5e4 1e5 2.5e5\
	   --defaults 1e-3 1e2 1e4 --ncols 3 --dpi 300\
	   --t1 10 --histories \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.06_cs_{}_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.06_cs_{}_nocsi_aux.pkl \
	   --name behaviour_of_tau_fcn_nocsi.png
	   
}

fmnist(){
    python comparison_poc_6.py --dataset fmnist --num-nodes 125 \
	   --c1 5e-5 7.5e-5 1e-4 2.5e-4 5e-4 7.5e-4 1e-3 2.5e-3 5e-3 7.5e-3 1e-2 \
	   --c2 5e0 7.5e0 1e1 2.5e1 5e1 7.5e1 1e2 2.5e2 5e2 7.5e2 1e3 \
	   --c3 7.5e2 1e3 2.5e3 5e3 7.5e3 1e4 2.5e4 5e4 7.5e4 1e5 2.5e5\
	   --defaults 1e-3 1e2 1e4 --ncols 3 --dpi 300\
	   --t1 10 --histories \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.06_cs_{}_nocsi.pkl \
	   clf_fcn_paradigm_hl_uniform_True_non_iid_1_num_workers_125_lr_0.0_decay_1e-05_batch_16_delta_10.0_zeta_0.001_beta_20.0_mu_1.0_phi_0.0_factor_2_T1_10_Tmax_40_E_0.06_D_0.06_cs_{}_nocsi_aux.pkl \
	   --name behaviour_of_tau_fcn_nocsi.png
	   
}


$1
