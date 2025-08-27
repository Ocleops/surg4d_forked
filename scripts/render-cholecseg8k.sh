########## exp setup ##########
export centers_num=3
export ONLY_EVAL=t
clip_feat_dim=3
video_feat_dim=6
dataset_name=video01_14939_final_for_training_cropped

########## time-agnostic language field ##########
export language_feature_hiddendim=${clip_feat_dim}
rm -rf submodules/4d-langsplat-rasterization/build 
pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
export use_discrete_lang_f=f
for level in 1 2 3; do
for mode in "lang" "rgb"; do
python render.py -s  data/cholecseg8k/${dataset_name} --model_path output/cholecseg8k/${dataset_name}/${dataset_name}_${level} --skip_train --skip_test --configs arguments/cholecseg8k/default.py --mode ${mode} --no_dlang 1 --load_stage fine-lang 
done
done

########## time-sensitive language field ##########
hiddendim=6
export language_feature_hiddendim=${video_feat_dim}
rm -rf submodules/4d-langsplat-rasterization/build 
pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
export use_discrete_lang_f=t
for level in 0; do
for mode in "lang" "rgb"; do
python render.py -s  data/cholecseg8k/${dataset_name} --model_path output/cholecseg8k/${dataset_name}/${dataset_name}_${level} --skip_train --skip_test --configs arguments/cholecseg8k/default.py --mode ${mode} --no_dlang 0 --load_stage fine-lang-discrete 
done
done

