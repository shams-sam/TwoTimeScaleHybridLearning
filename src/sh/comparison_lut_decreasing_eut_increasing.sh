python comparison.py --dataset mnist --num-nodes 125 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_0_radius_multi_d2d_1.0_factor_2_eut_20_lut_4_sigma_mul_0.01.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_0_radius_multi_d2d_1.0_factor_2_eut_30_lut_2_sigma_mul_0.01.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_0_radius_multi_d2d_1.0_factor_2_eut_40_lut_1_sigma_mul_0.01.pkl \
       --labels  'fl' 'eut=10, lut=0, $\theta$=50' 'eut=20, lut=4, $\theta$=*' 'eut=30, lut=2, $\theta$=*' 'eut=40, lut=1, $\theta$=*' \
       --ncols 3 --dpi 200\
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.003_decay_1e-05_batch_0_rounds_dyn_radius_multi_d2d_1.0_factor_2_lut_decreasing_eut_increasing.jpg


python comparison.py --dataset fmnist --num-nodes 125 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_0_radius_multi_d2d_1.0_factor_2_eut_20_lut_4_sigma_mul_0.01.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_0_radius_multi_d2d_1.0_factor_2_eut_30_lut_2_sigma_mul_0.01.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_0_radius_multi_d2d_1.0_factor_2_eut_40_lut_1_sigma_mul_0.01.pkl \
       --labels  'fl' 'eut=10, lut=0, $\theta$=50' 'eut=20, lut=4, $\theta$=*' 'eut=30, lut=2, $\theta$=*' 'eut=40, lut=1, $\theta$=*' \
       --ncols 3 --dpi 200\
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.002_decay_1e-05_batch_0_rounds_dyn_radius_multi_d2d_1.0_factor_2_lut_decreasing_eut_increasing.jpg
