from typing import Union

from torch.optim import Optimizer

from pytorch_lightning.accelerators.plugins.precision import PrecisionPlugin


class DeepSpeedPrecisionPlugin(PrecisionPlugin):

    def __init__(self, precision):
        super().__init__()
        self.precision = precision

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = float(2.0)):
        """
        DeepSpeed handles clipping gradients via the training type plugin. Override precision plugin
        to take no effect.
        """
        pass
