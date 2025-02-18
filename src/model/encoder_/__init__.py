from typing import Optional

from .encoder_ import Encoder_
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
}

EncoderCfg_ = EncoderNoPoSplatCfg


def get_encoder_(cfg: EncoderCfg_) -> tuple[Encoder_, Optional[EncoderVisualizer]]:
    encoder_, visualizer = ENCODERS[cfg.name]
    encoder_ = encoder_(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder_)
    return encoder_, visualizer
