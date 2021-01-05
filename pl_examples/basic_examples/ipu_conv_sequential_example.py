# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Experimental script to run a convolutional model using IPUs.
Currently not using real data but fake generated data, primarily for debugging.

python ipu_conv_sequential_example.py --max_epochs 1 --limit_val_batches 0

"""
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset
from torchvision.models import resnet18

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import IPUAccelerator
from pytorch_lightning.utilities import POPTORCH_AVAILABLE

if POPTORCH_AVAILABLE:
    import poptorch


class LitResnet(pl.LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = resnet18()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self._example_input_array = torch.randn((1, 3, 32, 32))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.02)
        return optimizer

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        return loss


class DummyImageClassificationDataset(Dataset):
    def __init__(self, pixels=224, num_samples: int = 10000, num_classes: int = 10):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        self.num_samples = num_samples
        self.pixels = pixels
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = torch.rand(3, self.pixels, self.pixels)
        y = torch.randint(0, self.num_classes, size=(1,))[0]
        return [x, y]


def instantiate_dataset(ipu_opts, batch_size, num_workers):
    ds = DummyImageClassificationDataset(pixels=224, num_classes=10)

    # todo: we should autoswap torch dataloaders to this type within the accelerator.
    loader = poptorch.DataLoader(ipu_opts, ds, batch_size=batch_size, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    cli_lightning_logo()
    parser = ArgumentParser(description="IPU Convolution Example")
    parser = Trainer.add_argparse_args(parser)
    parser = IPUAccelerator.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=2, type=int)
    args = parser.parse_args()

    training_opts, inference_opts = IPUAccelerator.parse_opts(args)

    train_loader = instantiate_dataset(ipu_opts=training_opts, batch_size=args.batch_size, num_workers=4)

    accelerator = IPUAccelerator(training_opts, inference_opts)

    model = LitResnet()

    trainer = pl.Trainer.from_argparse_args(args, accelerator=accelerator)
    trainer.fit(model, train_loader)
