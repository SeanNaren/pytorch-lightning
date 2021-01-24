from typing import Union

from torch.optim import Optimizer

from pytorch_lightning.accelerators.plugins.precision import PrecisionPlugin


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """
    DeepSpeed handles all precision internally via the TrainingType Plugin.
    This class prevents any unexpected precision behaviour from occuring.
    """
    precision = 32

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = float(2.0)):
        """
        DeepSpeed handles clipping gradients via the training type plugin. Override precision plugin
        to take no effect.
        """
        pass
