python train_unet.py \
  -b 32 \
  -e 5 \
  -l 0.0005 \
  -r 5 \
  -n 'test_Unet+NAFnet' \
  -t '/home/Data/train/image/' \
  -v '/home/Data/val/image/' \
  --seed 91