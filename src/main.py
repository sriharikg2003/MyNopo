import os
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
import random

import torch.nn.init as init

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.encoder_ import get_encoder_
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)


    # SAM
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    from hydra.core.global_hydra import GlobalHydra

  

    # This allows the current step to be shared with the data loader processes.


    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    encoder_, encoder_visualizer_ = get_encoder_(cfg.model.encoder_)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.

    # if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
    #     weight_path = cfg.model.encoder.pretrained_weights
    #     ckpt_weights = torch.load(weight_path, map_location='cpu')
    #     if 'model' in ckpt_weights:
    #         print('model load')
            
    #         ckpt_weights = ckpt_weights['model']
    #         ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)


    #         missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
    #     elif 'state_dict' in ckpt_weights:

    #         print('state_dict load')
    #         ckpt_weights = ckpt_weights['state_dict']
    #         ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
    #         missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
    #     else:
    #         raise ValueError(f"Invalid checkpoint format: {weight_path}")
    #     exit()

    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')


        excluded_keys = [
            "downstream_head1.dpt.head.0.weight",
            "downstream_head1.dpt.head.0.bias",
            "downstream_head2.dpt.head.0.weight",
            "downstream_head2.dpt.head.0.bias",
            "gaussian_param_head.dpt.head.0.weight",
            "gaussian_param_head2.dpt.head.0.weight",
            "gaussian_param_head.dpt.head.0.bias"
            "gaussian_param_head2.dpt.head.0.bias"

        ]


        if 'model' in ckpt_weights:

            print('model load')
            
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            
            model_state_dict = encoder.state_dict()
            
            new_ckpt_weights = {}
            for key, value in ckpt_weights.items():
                if key in excluded_keys:
                    if key in model_state_dict:  
                        expected_shape = model_state_dict[key].shape
                        print(f"Initializing {key} with random values of shape {expected_shape}")
                        
                        if len(expected_shape) >= 2:  
                            new_ckpt_weights[key] = torch.nn.init.xavier_uniform_(torch.empty(expected_shape))
                        else: 
                            new_ckpt_weights[key] = torch.zeros(expected_shape)
                    else:
                        print(f"Skipping {key} as it does not exist in model.")
                else:
                    new_ckpt_weights[key] = value

          
            missing_keys, unexpected_keys = encoder.load_state_dict(new_ckpt_weights, strict=False)



            # Original Noposplat

            ckpt_weights_ = ckpt_weights_['model']
            ckpt_weights_ = checkpoint_filter_fn(ckpt_weights_, encoder_)
            
            model_state_dict_ = encoder_.state_dict()
                     
            encoder_.load_state_dict(model_state_dict_, strict=False)
            
        elif 'state_dict' in ckpt_weights:

            print('Loading state_dict')

            # Load the checkpoint weights
            ckpt_weights = ckpt_weights['state_dict']

            # Separate the weights for encoder_ and encoder
            encoder_Weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}

            
            model_state_ = encoder_.state_dict()

            # Loop over the encoder_ weights and load them
            for k, v in encoder_Weights.items():
                if k in model_state_:
                    if model_state_[k].shape == v.shape:
                        model_state_[k].copy_(v)  # Directly copy weights if shape matches
                    else:
                        print(f"Initializing {k} due to shape mismatch: {v.shape} vs {model_state_[k].shape}")
                        exit()
                else:
                    print(f"Skipping {k} because it is not found in encoder_ model state")

            # Apply the state dict to encoder_ model
            encoder_.load_state_dict(model_state_, strict=False)

            # Now for encoder (not encoder_)
            encoder_Weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            model_state = encoder.state_dict()

            # Loop over the encoder weights and load them
            for k, v in encoder_Weights.items():
                if k in model_state:
                    if model_state[k].shape == v.shape:
                        model_state[k].copy_(v)  # Directly copy weights if shape matches
                    else:
                        print(f"Initializing {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                        print("********")
                        param = model_state[k]
                        if param.dim() > 1:  
                            init.kaiming_normal_(param)  # Initialize conv/linear layers
                        else:
                            init.zeros_(param)  # Initialize biases to zero
                else:
                    print(f"Skipping {k} because it is not found in encoder model state")

            # Apply the state dict to encoder model
            encoder.load_state_dict(model_state, strict=False)







        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")
            
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path )
    else:

        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
            
        )



if __name__ == "__main__":


    train()

