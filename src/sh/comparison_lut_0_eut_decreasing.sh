python comparison.py --dataset mnist --num-nodes 125 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0_eut_gamma_0.80.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0_eut_gamma_1.20.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0_eut_gamma_0.80.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0_eut_gamma_1.20.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_40_lut_0_eut_gamma_0.80.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_40_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_40_lut_0_eut_gamma_1.20.pkl \
       --labels  'fl' 'eut=10, $\Gamma$=0.8' 'eut=10, $\Gamma$=1.0' 'eut=10, $\Gamma$=1.2' 'eut=20, $\Gamma$=0.8' 'eut=20, $\Gamma$=1.0' 'eut=20, $\Gamma$=1.2' 'eut=40, $\Gamma$=0.8' 'eut=40, $\Gamma$=1.0' 'eut=40, $\Gamma$=1.2' \
       --ncols 5 --dpi 200\
       --colors 'ko-' 'rx:' 'r--' 'r.-.' 'bx:' 'b--' 'b.-.' 'gx:' 'g--' 'g.-.' \
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_dyn_radius_multi_d2d_1.0_factor_2_lut_0_eut_period_varying.jpg


python comparison.py --dataset fmnist --num-nodes 125 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0_eut_gamma_0.80.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0_eut_gamma_1.20.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0_eut_gamma_0.80.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0_eut_gamma_1.20.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_40_lut_0_eut_gamma_0.80.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_40_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_40_lut_0_eut_gamma_1.20.pkl \
       --labels  'fl' 'eut=10, $\Gamma$=0.8' 'eut=10, $\Gamma$=1.0' 'eut=10, $\Gamma$=1.2' 'eut=20, $\Gamma$=0.8' 'eut=20, $\Gamma$=1.0' 'eut=20, $\Gamma$=1.2' 'eut=40, $\Gamma$=0.8' 'eut=40, $\Gamma$=1.0' 'eut=40, $\Gamma$=1.2' \
       --ncols 5 --dpi 200\
       --colors 'ko-' 'rx:' 'r--' 'r.-.' 'bx:' 'b--' 'b.-.' 'gx:' 'g--' 'g.-.' \
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_dyn_radius_multi_d2d_1.0_factor_2_lut_0_eut_period_varying.jpg
