cd autoencoder
## train autoencoder for clip features ###
echo "Training autoencoder with clip features"
dataset_name=video01_28820
dataset_path=../data/cholecseg8k/video01/${dataset_name}
clip_feature_name=language_features_default
# video_feature_name=video_features
clip_dim=3
# video_dim=6
python train.py --lr 7e-4 --dataset_path ${dataset_path} --model_name ${dataset_name}_clip --feature_dims 512  \
    --encoder_dims 256 128 64 32 ${clip_dim} --decoder_dims 16 32 64 128 256 512 --hidden_dims ${clip_dim} --language_name ${clip_feature_name}

python test.py --dataset_path ${dataset_path} --model_name ${dataset_name}_clip --feature_dims 512 \
    --encoder_dims 256 128 64 32 ${clip_dim} --decoder_dims 16 32 64 128 256 512 --hidden_dims ${clip_dim} --language_name ${clip_feature_name}

### train autoencoder for video features ###
# echo "Training autoencoder with video features"
# python train.py --lr 7e-5 --dataset_path ${dataset_path} --model_name ${dataset_name}_video --feature_dims 4096  \
#     --encoder_dims 2048 1024 512 256 128 64 32 ${video_dim} --decoder_dims 32 64 128 256 512 1024 2048 4096 --hidden_dims ${video_dim} --cos_weight 1e-2 --language_name ${video_feature_name}

# python test.py --dataset_path ${dataset_path} --model_name  ${dataset_name}_video --feature_dims 4096 \
#     --encoder_dims 2048 1024 512 256 128 64 32 ${video_dim} --decoder_dims 32 64 128 256 512 1024 2048 4096 --hidden_dims ${video_dim} --language_name ${video_feature_name}
