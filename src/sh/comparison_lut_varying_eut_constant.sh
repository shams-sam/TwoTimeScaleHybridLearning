# python comparison.py --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_1.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_2.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_5.pkl \
#        --labels  'fl' 'eut=10, lut=1' 'eut=10, lut=2' 'eut=10, lut=5' --ncols 2\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_varying_eut_10.jpg


# python comparison.py --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_1.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_2.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_4.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_5.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_10.pkl \
#        --labels  'fl' 'eut=10, lut=1' 'eut=10, lut=2' 'eut=10, lut=4' 'eut=10, lut=5' 'eut=10, lut=10' --ncols 3\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_varying_eut_20.jpg

# python comparison.py --dataset mnist --num-nodes 25 --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_1.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_2.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_3.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_5.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_6.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_10.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_15.pkl \
#        --labels  'fl' 'eut=30, lut=1' 'eut=30, lut=2' 'eut=30, lut=3' 'eut=30, lut=5' 'eut=30, lut=6' 'eut=30, lut=10' 'eut=30, lut=15' \
#        --ncols 4 --dpi 200\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.006_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_varying_eut_30.jpg

# python comparison.py --dataset mnist --num-nodes 25 --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_1.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_2.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_5.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_10.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_15.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_30.pkl \
#        --labels  'fl' 'eut=30, lut=0' 'eut=30, lut=1' 'eut=30, lut=2' 'eut=30, lut=5' 'eut=30, lut=10' 'eut=30, lut=15' 'eut=30, lut=30' \
#        --ncols 4 --dpi 200\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_varying_eut_30.jpg


python comparison.py --dataset fmnist --num-nodes 25 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_1.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_2.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_5.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_10.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_15.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_30.pkl \
       --labels  'fl' 'eut=30, lut=0' 'eut=30, lut=1' 'eut=30, lut=2' 'eut=30, lut=5' 'eut=30, lut=10' 'eut=30, lut=15' 'eut=30, lut=30' \
       --ncols 4 --dpi 200\
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_varying_eut_30.jpg
