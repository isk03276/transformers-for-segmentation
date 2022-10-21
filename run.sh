# unetr
#python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 800 --save-interval 50 --model-name unetr --model-config-file configs/unetr/default_unetr.yaml

#deformable unetr - pw
#python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 500 --save-interval 50 --model-name deformable_unetr --model-config-file configs/deformable_unetr/default_deformable_unetr_pw.yaml

#deformable unetr - no pw
python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 800 --save-interval 50 --model-name deformable_unetr --model-config-file configs/deformable_unetr/default_deformable_unetr.yaml

#deformable unetr - no static positional encoding
#python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 800 --save-interval 50 --model-name deformable_unetr --model-config-file configs/deformable_unetr/default_deformable_unetr_no_static_positional_encoding.yaml

#deformable unetr - no dynamic positional encoding
python main.py --batch-size 2 --validation-set-ratio 0.1 --epoch 800 --save-interval 50 --model-name deformable_unetr --model-config-file configs/deformable_unetr/default_deformable_unetr_no_dynamic_positional_encoding.yaml