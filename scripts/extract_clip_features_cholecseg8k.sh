dataset_path=/home/tumai/team1/Ken/4DLangSplatSurgery/data/cholecseg8k/video01/video01_14939_firstry
precompute_seg_path=/home/tumai/team1/Ken/4DLangSplatSurgery/submodules/4d-langsplat-tracking-anything-with-deva/output/default/origin_mask_default
clip_language_feature_name=clip_features
cd preprocess
python generate_clip_features_cholecseg8k.py --dataset_path $dataset_path \
  --dataset_type hypernerf \
  --precompute_seg ${precompute_seg_path} \
  --output_name ${clip_language_feature_name}

 