## Inference

0. Prepare Pre-trained models
   You can download the pre-trained [stylegan model](https://drive.google.com/file/d/14-Sv793VyBrSD-xXefCMJ_WYgpXIkv35/view?usp=sharing) and [vessel-seg model](https://drive.google.com/file/d/1qXn_JvYr3bk10_PTUdonwUGvgb_OU87a/view?usp=sharing), then put them in 'weights' folder.
1. Get latent code

   ```shell
   python projector.py --outdir=out --target=fundus.png --network=weights/network-snapshot-005400.pkl
   ```

    or you can use[e4e](https://github.com/omertov/encoder4editing) to generate latent code. (Reconstruction will be much better than projector.py)

2. Modify the code in deid/scripts/edit.py

   ```
   z_filename = 'your_latent_code.npy'
   ```
3. Change the Segmentation method

   ```
   if use_vessel_mask:
   	mask = vessel_seg(image)
   if others:
   	mask = other_seg(image) # replace with your own segmentation method
   ```
4. Set select_mode

   ```
   select_mode = 0 # select top n based on the score
   select_mode = 1 # chosen in the order provided by the list
   select_mode = 2 # manually set the layers and channels.
   ```

## Training

Refer to [Stylegan3](https://github.com/NVlabs/stylegan3)

```shell
#### Config
https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md
### dataset
python dataset_tool.py --source=/tmp/images --dest=~/datasets/ffhq-256x256.zip \
    --resolution=256x256
### train
CUDA_VISIBLE_DEVICES=2,3,1,5  python train.py --outdir training-runs --cfg=stylegan3-t --data data/debatic.zip --batch 64 --gpus 4 --resume training-runs/00007-stylegan3-t-debatic-gpus4-batch64-gamma8.2/network-snapshot-001814.pkl --gamma=8.2 --mirror=1

```
