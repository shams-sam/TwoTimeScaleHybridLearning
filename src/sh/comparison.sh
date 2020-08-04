# # iid dataset comparison of different lut intervals for a constant lut
# python comparison.py --histories \
#        clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_10.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_8.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_6.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_4.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_2.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_1.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        --labels  'fl' 'eut=10,lut=10' 'eut=10,lut=8' 'eut=10,lut=6' 'eut=10,lut=4' 'eut=10,lut=2' 'eut=10,lut=1' 'eut=10 w/ avg' --ncols 4 \
#        --name comparison_clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_lut.jpg

# # non-iid dataset comparison of different lut intervals for a constant lut
# python comparison.py --histories \
#        clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_10.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_8.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_6.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_4.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_2.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_1.pkl \
#        clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        --labels  'fl' 'eut=10,lut=10' 'eut=10,lut=8' 'eut=10,lut=6' 'eut=10,lut=4' 'eut=10,lut=2' 'eut=10,lut=1' 'eut=10 w/ avg' --ncols 4 \
#        --name comparison_clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_lut.jpg

# iid dataset comparison of different number of rounds for constant lut and eut
python comparison.py --histories \
       clf_fcn_paradigm_fl_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_1_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_5_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_15_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_30_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       --labels  'fl' '$\theta$=1' '$\theta$=5' '$\theta$=15' '$\theta$=30' '$\theta$=50' --ncols 3 \
       --name comparison_clf_fcn_paradigm_fog_uniform_True_non_iid_10_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_rounds.jpg

# non iid dataset comparison of different number of rounds for constant lut and eut
python comparison.py --histories \
       clf_fcn_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_1_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_5_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_15_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_30_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
       --labels  'fl' '$\theta$=1' '$\theta$=5' '$\theta$=15' '$\theta$=30' '$\theta$=50' --ncols 3\
       --name comparison_clf_fcn_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.01_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_rounds.jpg
