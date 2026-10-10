# Model weights

Place the DINOv3 ConvNeXt Tiny pretrained checkpoint here:

`dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth`

This checkpoint was copied from the existing local Torch cache. Weight files are
gitignored; provide this file separately when setting up another checkout.
DINOv3's source and license are in the pinned submodule at `src/dinov3/`.

Training starts from this file. Saved Lightning checkpoints contain the encoder
and detection weights, so prediction does not require the original `.pth` file.
