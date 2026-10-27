import os
from pathlib import Path
import warnings

import hydra
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# Configure beartype and jaxtyping.
# with install_import_hook(
#     ("src",),
#     ("beartype", "beartype"),
# ):
from transplat.src.config import load_typed_root_config
from transplat.src.dataset.data_module import DataModule
from transplat.src.global_cfg import set_cfg
from transplat.src.loss import get_losses
from transplat.src.misc.LocalLogger import LocalLogger
from transplat.src.misc.step_tracker import StepTracker
from model_wrapper import ModelWrapper
# from decoder import get_decoder
from gaussseed_encoder import get_encoder
from transplat.src.model.decoder import get_decoder
# from transplat.src.model.encoder import get_encoder


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

#     使用 Hydra 的 @hydra.main 装饰器定义主函数 train。
#     config_path 和 config_name 指定了配置文件的路径和名称。
#     Hydra 会加载配置文件，并将其内容注入到 cfg_dict 参数中。
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)


def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)#加载并验证配置文件的内容。
    set_cfg(cfg_dict)#将配置文件的内容设置到全局变量中

    # Set up the output directory.设置输出目录
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    latest_run.unlink(missing_ok=True)
    latest_run.symlink_to(output_dir)

    # Keep review-package training local and free of account-specific metadata.
    callbacks = []
    logger = LocalLogger()

    # Set up checkpointing.设置检查点保存
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_last=True,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.准备检查点加载
    checkpoint_path = (
        None
        if cfg.checkpointing.load is None
        else Path(cfg.checkpointing.load).expanduser().resolve()
    )

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
    # 初始化训练器
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    

    # 初始化模型和数据模块
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker
    )
    
    # Freeze all parameters except pose head
    # for param in model_wrapper.parameters():
    #     param.requires_grad = False
    # for param in model_wrapper.encoder.backbone_t.pose_head.parameters():
    #     param.requires_grad = True
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 加载除pose_head外的所有权重
        model_state = model_wrapper.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() 
        #                 if not k.startswith('encoder.backbone.pose_head')}
        
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                        }
        
        model_state.update(pretrained_dict)
        model_wrapper.load_state_dict(model_state, strict=False)
        
        # 初始化pose_head的新参数
        # for m in model_wrapper.encoder.backbone.pose_head.modules():
        #     if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             torch.nn.init.constant_(m.bias, 0)
        
        print(cyan(f"Reinitialized pose_head and loaded other weights"))

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
                



    # 训练或测试
    if cfg.mode == "train":
        torch.autograd.set_detect_anomaly(True)  # 放在训练代码开头

            #         # 对于冻结层的输出，设置不需要梯度
            #         for mod_name, module in model_wrapper.named_modules():
            #             if mod_name in name:
            #                 module.requires_grad_(False)

        # 开始训练（不使用 ckpt_path 参数）
        trainer.fit(model_wrapper, datamodule=data_module)
        if trainer.is_global_zero:
            final_ckpt = output_dir / "checkpoints" / "last.ckpt"
            final_ckpt.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(final_ckpt)
            print(cyan(f"Saved final checkpoint to {final_ckpt}."))
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")#忽略警告信息。
    torch.set_float32_matmul_precision('high')#设置 PyTorch 的矩阵乘法精度为高。

    train()
