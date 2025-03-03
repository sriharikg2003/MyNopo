from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
