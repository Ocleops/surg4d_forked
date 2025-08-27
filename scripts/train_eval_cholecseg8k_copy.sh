########## exp setup ##########
export centers_num=3
clip_feat_dim=3
video_feat_dim=6
dataset_name=video01_14939_final_for_training_cropped_editedpointcloud
language_feature_name=language_features_pca3

########## time-agnostic language field ##########
export language_feature_hiddendim=${clip_feat_dim}
# rm -rf submodules/4d-langsplat-rasterization/build && pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
export use_discrete_lang_f=f
# for level in 1 2 3; do
level=0
python train_cholecseg8k.py --source_path data/cholecseg8k/${dataset_name} --port 6021 --expname cholecseg8k/${dataset_name}/${dataset_name}_${level} --configs arguments/hypernerf/default.py --include_feature \
    --language_features_name ${language_feature_name} --feature_level ${level} --joint_coarse --no_dlang 1
for mode in "lang" "rgb"; do
python render.py -s  data/hypernerf/${dataset_name} --language_features_name ${language_feature_name}-language_features_dim${clip_feat_dim} --model_path output/hypernerf/${dataset_name}/${dataset_name}_${level} \
    --feature_level ${level} --skip_train --skip_test --configs arguments/hypernerf/default.py --mode ${mode} --no_dlang 1 --load_stage fine-lang 
done
# done

# ########## time-sensitive language field ##########
# level=0
# language_feature_name=video_features
# export language_feature_hiddendim=${video_feat_dim}
# rm -rf submodules/4d-langsplat-rasterization/build 
# pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
# export use_discrete_lang_f=f
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname hypernerf/${dataset_name}/${dataset_name}_${level} --configs arguments/hypernerf/chicken.py --include_feature \
#     --language_features_name ${language_feature_name}-language_features_dim${video_feat_dim} --feature_level ${level} --fine_lang_iterations 0 --joint_coarse --no_dlang 0 --checkpoint_iterations 10000

# export use_discrete_lang_f=t
# python train.py -s data/hypernerf/${dataset_name} --port 6021 --expname hypernerf/${dataset_name}/${dataset_name}_${level} --configs arguments/hypernerf/chicken.py --include_feature \
#     --language_features_name ${language_feature_name}-language_features_dim${video_feat_dim} --feature_level ${level} --joint_coarse --no_dlang 0 --resume_from_final_stage 1 --start_checkpoint output/hypernerf/${dataset_name}/${dataset_name}_${level}/chkpnt_fine-base_10000.pth

# for mode in "lang" "rgb"; do
# python render.py -s  data/hypernerf/${dataset_name} --feature_level ${level} --language_features_name ${language_feature_name}-language_features_dim${video_feat_dim} \
#     --model_path output/hypernerf/${dataset_name}/${dataset_name}_${level} --skip_train --skip_test --configs arguments/hypernerf/chicken.py --mode ${mode} --no_dlang 0 --load_stage fine-lang-discrete 
# done

# ########## Evaluate ##########
# cd eval
# python eval.py --dataset_type hypernerf \
#     --annotation_folder ../data/hypernerf/${dataset_name}/annotations \
#     --exp_name ${dataset_name}/${dataset_name} \
#     --feat_dim ${clip_feat_dim} \
#     --video_feat_dim ${video_feat_dim} \
#     --iterations 10000 \
#     --video_eval_iterations 20000 \
#     --ae_ckpt_path ../autoencoder/ckpt/${dataset_name}_clip/best_ckpt.pth \
#     --video_ae_ckpt_path ../autoencoder/ckpt/${dataset_name}_video/best_ckpt.pth \
#     --video_feat_dir ${dataset_name}/${dataset_name} \
#     --apply_video_search \
#     --smooth_feature_post