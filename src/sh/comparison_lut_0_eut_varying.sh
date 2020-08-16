# non iid dataset comparison
# clfs: svm
# dataset: mnist
# 1. fog with increasing delay among euts and no luts
# 2. addition of lut rounds to see the effect on the convergence
# python comparison.py --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_2_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_5_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_15_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
#        --labels  'fl' 'eut=2, lut=0' 'eut=5, lut=0' 'eut=10, lut=0' 'eut_15, lut=0' 'eut=20, lut=0' '$\theta$=50' --ncols 3\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_0_eut_varying.jpg

# python comparison.py --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_2_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_50_lut_0.pkl \
#        --labels  'fl' 'eut=2, lut=0' 'eut=10, lut=0' 'eut=20, lut=0' 'eut=30, lut=0', 'eut=50, lut=0' \
#        --ncols 3 --dpi 200\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.007_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_0_eut_varying.jpg

# python comparison.py --dataset mnist --num-nodes 25 --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_2_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_50_lut_0.pkl \
#        --labels  'fl' 'eut=2, lut=0' 'eut=10, lut=0' 'eut=20, lut=0' 'eut=30, lut=0', 'eut=50, lut=0' \
#        --ncols 3 --dpi 200\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.003_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_0_eut_varying.jpg


# non iid dataset comparison
# clfs: svm
# dataset: fmnist
# 1. fog with increasing delay among euts and no luts
# 2. addition of lut rounds to see the effect on the convergence
# python comparison.py --dataset fmnist --num-nodes 125 --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_2_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_5_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_15_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
#        --labels  'fl' 'eut=2, lut=0' 'eut=5, lut=0' 'eut=10, lut=0' 'eut_15, lut=0' 'eut=20, lut=0' \
#        --ncols 3\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.001_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_0_eut_varying.jpg

# python comparison.py --dataset fmnist --num-nodes 125 --histories \
#        clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_2_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_0.pkl \
#        clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_50_lut_0.pkl \
#        --labels  'fl' 'eut=2, lut=0' 'eut=10, lut=0' 'eut_20, lut=0' 'eut=30, lut=0' 'eut=50, lut=0' \
#        --ncols 3 --dpi 200\
#        --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_125_lr_0.0035_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_0_eut_varying.jpg

python comparison.py --dataset fmnist --num-nodes 25 --histories \
       clf_svm_paradigm_fl_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_2_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_10_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_20_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_30_lut_0.pkl \
       clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_eut_50_lut_0.pkl \
       --labels  'fl' 'eut=2, lut=0' 'eut=10, lut=0' 'eut=20, lut=0' 'eut=30, lut=0', 'eut=50, lut=0' \
       --ncols 3 --dpi 200\
       --name comparison_clf_svm_paradigm_fog_uniform_True_non_iid_1_num_workers_25_lr_0.002_decay_1e-05_batch_0_rounds_50_radius_multi_d2d_1.0_factor_2_lut_0_eut_varying.jpg
