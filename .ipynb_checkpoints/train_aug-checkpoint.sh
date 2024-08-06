## delay: n번재 epoch까지는 쉼
python train_aug.py \
  -b 1 \
  -e 2 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet' \
  -t '/home/Data/train/' \
  -v '/home/Data/val/' \
  --cascade 11 \
  --chans 18 \
  --sens_chans 8 \
  --seed 91 \
  --aug_delay 0 \
  --aug_strength 0.5