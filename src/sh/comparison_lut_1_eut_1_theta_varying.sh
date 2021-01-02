python comparison.py --dataset mnist --num-nodes 125 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_1_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_2_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_4_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_8_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_16_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_32_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_64_radius_multi_d2d_1.0_factor_2_eut_1_lut_1.pkl \
       --labels  'fl' 'eut,lut=1, $\theta$=1' 'eut,lut=1, $\theta$=2' 'eut,lut=1, $\theta$=4' 'eut,lut=1, $\theta$=8' 'eut,lut=1, $\theta$=16' 'eut,lut=1, $\theta$=32' 'eut,lut=1, $\theta$=64' \
       --ncols 4 --dpi 200\
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.006_decay_1e-05_batch_0_rounds_varying_radius_multi_d2d_1.0_factor_2_lut_1_eut_1.jpg
