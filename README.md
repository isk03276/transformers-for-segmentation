# transformers-for-segmentation

Transformers for 3d image segmentation
- Models
  - UnetR
  - UnetR + deformable-attention

- DRL framework : PyTorch
- Dataset : [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)

## Install
```bash
git clone https://github.com/isk03276/transformers-for-segmentation
cd transformers-for-segmentation
pip install -r requirements.txt
```
## Getting Started
```bash
python main.py --batch-size [BATCH-SIZE] --model-name [MODEL-NAME(ex. 'unetr', 'patch-wise-deformable-attention')] --n-folds 5 #train
python main.py --test
```

## Results
**- BTCV**  
<img src="https://user-images.githubusercontent.com/23740495/197909016-0f532e3f-588c-44b1-93fc-692fffa6ef20.png" width="100%" height="100%"/>

## References
- [UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504)
- [Vision Transformer with Deformable Attention](https://arxiv.org/abs/2201.00520)
