## for low-resolution (256 * 256)

python train.py models/DucoNet_256.py --workers=8 --gpus=0,1 --exp-name=DucoNet_256 --batch-size=32

## for high-resolution (1024 * 1024)

#python train.py models/DucoNet_1024.py --workers=8 --gpus=0,1 --exp-name=DucoNet_1024 --batch-size=4