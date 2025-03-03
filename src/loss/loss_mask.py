from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMaskCfg:
    weight: float


@dataclass
class LossMaskCfgWrapper:
    mask: LossMaskCfg


class LossMask(Loss[LossMaskCfg, LossMaskCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.gaussians.means - batch["context"]["rep"]
        return self.cfg.weight * (delta**2).mean()
