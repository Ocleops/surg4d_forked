dataset_path=data/hypernerf/chickchicken
precompute_seg_path=submodules/4d-langsplat-tracking-anything-with-deva/output_devaseg/video_mask_concat
clip_language_feature_name=clip_features
cd preprocess
python generate_clip_features.py --dataset_path ../$dataset_path \
  --dataset_type hypernerf \
  --precompute_seg ../${precompute_seg_path} \
  --output_name ${clip_language_feature_name}
