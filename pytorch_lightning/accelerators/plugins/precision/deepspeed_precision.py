from typing import Union

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Optimizer

from pytorch_lightning.accelerators.plugins.precision import PrecisionPlugin


class DeepSpeedPrecisionPlugin(PrecisionPlugin):

    def __init__(self, precision):
        super().__init__()
        self.precision = precision

    def backward(
        self,
        lightning_module: LightningModule,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ):
        # todo a hack around so that the model itself can run backwards...
        # todo this also means that the lightning module backward function is never called which may be unfavourable
        deepspeed_engine = lightning_module.trainer.model
        deepspeed_engine.backward(closure_loss)
        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = float(2.0)):
        """
        DeepSpeed handles clipping gradients via the training type plugin. Override precision plugin
        to take no effect.
        """
        pass
