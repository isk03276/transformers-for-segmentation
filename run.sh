#deformable unetr - pw
python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 300 --save-interval 100 --model-name patch_wise_deformable_unetr --model-config-file configs/patch_wise_deformable_unetr/default.yaml --num-classes 14 --dataset-name occluded_btcv --load-from checkpoints/btcv/patch_wise_deformable_unetr/epoch_400

# unetr
python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 300 --save-interval 100 --model-name unetr --model-config-file configs/unetr/default.yaml --num-classes 14 --dataset-name occluded_btcv --load-from checkpoints/btcv/unetr/epoch_400

#deformable unetr - no pw
python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 300 --save-interval 100 --model-name deformable_unetr --model-config-file configs/deformable_unetr/default.yaml --num-classes 14 --dataset-name occluded_btcv --load-from checkpoints/btcv/deformable_unetr/epoch_400