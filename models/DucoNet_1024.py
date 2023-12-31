from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer
from iharm.model import initializer
from iharm.model.base import DucoNet_model
from iharm.model.losses import MaskWeightedMSE
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric
from iharm.utils.log import logger


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)

def _model_init(model,_skip_init_names,cnt,max_cnt=4):
    if cnt>max_cnt:
        return
    for name,module in model.named_children():
        if name in _skip_init_names[cnt]:
            # print("skip:",name)
            _model_init(module,_skip_init_names,cnt+1,max_cnt=max_cnt)
        else:
            # print(name)
            module.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (1024,1024)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }
    model_cfg.depth = 4

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])

    model = DucoNet_model(
        depth=4, ch=32, image_fusion=True, attention_mid_k=0.5,
        attend_from=2, batchnorm_from=2,w_dim=256,control_module_start=cfg.control_module_start
    )

    model.to(cfg.device)

    _skip_init_names={
        0:['decoder'],
        1:['up_blocks'],
        2:['0','1','2'],
        3:['control_module'],
        4:['a_styleblock','b_styleblock','l_styleblock']
    }

    cnt=0
    _model_init(model,_skip_init_names,cnt,max_cnt=4)

    return model, model_cfg


def train(model, cfg, model_cfg, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeightedMSE(min_area=100)
    loss_cfg.pixel_loss_weight = 1.0

    num_epochs = 120

    train_augmentator = HCompose([
        RandomResizedCrop(*crop_size, scale=(0.5, 1.0)),
        HorizontalFlip(),
    ])

    val_augmentator = HCompose([
        Resize(*crop_size)
    ])


    trainset = ComposeDataset(
        [
            # HDataset(cfg.HFLICKR_PATH, split='train'),
            # HDataset(cfg.HDAY2NIGHT_PATH, split='train'),
            # HDataset(cfg.HCOCO_PATH, split='train'),
            HDataset(cfg.HADOBE5K1_PATH, split='train'),
        ],
        augmentator=train_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05,
    )

    valset = ComposeDataset(
        [
            # HDataset(cfg.HFLICKR_PATH, split='test'),
            # HDataset(cfg.HDAY2NIGHT_PATH, split='test'),
            # HDataset(cfg.HCOCO_PATH, split='test'),
            HDataset(cfg.HADOBE5K1_PATH, split='test'),
        ],
        augmentator=val_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1,
    )

    optimizer_params = {
        'lr': cfg.lr,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }
    lr_g = (cfg.batch_size / 64) ** 0.5
    optimizer_params['lr'] = optimizer_params['lr'] * lr_g

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[105, 115], gamma=0.1)
    trainer = SimpleHTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[
            DenormalizedPSNRMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            ),
            DenormalizedMSEMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            )
        ],
        checkpoint_interval=10,
        image_dump_interval=1000
    )

    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
