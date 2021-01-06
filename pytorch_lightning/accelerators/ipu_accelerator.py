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
import json
import os
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators.accelerator import Accelerator, ReduceOp
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import POPTORCH_AVAILABLE, rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if POPTORCH_AVAILABLE:
    import poptorch

    if not poptorch.ipuHardwareIsAvailable():
        raise MisconfigurationException("IPU Accelerator requires IPU hardware to run.")


class IPUOptsBuilder:
    """
    Class to build IPU opts based on poptorch options.
    This makes it easier to instantiate new opts from a shared set of arguments, and enforce
    options for inference.
    """

    def __init__(self, args):
        self.num_device_iterations = args.num_device_iterations
        self.replication_factor = args.replication_factor
        self.gradient_accumulation = args.gradient_accumulation
        self.auto_round_num_ipus = args.auto_round_num_ipus

    @property
    def training_opts(self):
        return self._create_opts(gradient_accumulation=self.gradient_accumulation)

    @property
    def inference_opts(self):
        return self._create_opts(gradient_accumulation=1)

    def _create_opts(self, gradient_accumulation):
        opts = poptorch.Options()
        opts.deviceIterations(self.num_device_iterations)
        opts.replicationFactor(self.replication_factor)
        opts.Training.gradientAccumulation(gradient_accumulation)
        opts.autoRoundNumIPUs(self.auto_round_num_ipus)
        return opts

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument(
            '--gradient_accumulation',
            default=1,
            type=int,
            help="Accumulate gradients. When using pipelining/model splitting it's a good idea to set this higher"
                 "to allow better parallelization within the sequential pipeline."
        )
        parser.add_argument(
            '--num_device_iterations',
            default=1,
            type=int,
            help="Number of iterations to run directly on one IPU before returning. "
                 "Improves Efficiency since loop runs directly on the IPU."
        )
        parser.add_argument(
            '--replication_factor',
            default=1,
            type=int,
            help="Defines the number of models to train in data parallel"
        )
        parser.add_argument(
            '--auto_round_num_ipus',
            action='store_true',
            help="Allocate power of 2 IPUs regardless if the model doesn't utilize all IPUs. This is a hard"
                 "requirement of reserving IPUs."
        )
        return parser


class IPUDebugOpts:
    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument(
            '--autoreport',
            action='store_true',
            help="Enable IPU reporting."
        )
        parser.add_argument(
            '--autoreport_dir',
            default=None,
            type=str,
            help="Save directory for logging, by default this uses the current working directory."
        )
        return parser

    @classmethod
    def parse_environment_debug_opts(cls, args):
        if args.autoreport:
            options = {
                "autoReport.all": args.autoreport
            }
            if args.autoreport_dir:
                if not os.path.exists(args.autoreport_dir):
                    os.makedirs(args.autoreport_dir)
                options["autoReport.directory"] = args.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)


class IPUAccelerator(Accelerator):

    def __init__(self,
                 training_opts,
                 inference_opts,
                 trainer=None,
                 mixed_precision: bool = False,
                 half: bool = False,
                 cluster_environment: Optional[ClusterEnvironment] = None):
        """
        Runs training using a single GPU

        Example::

            # default
            trainer = Trainer(accelerator=GPUAccelerator())

        """
        super().__init__(trainer, cluster_environment)
        self.mixed_precision = mixed_precision
        self.half = half
        self.training_opts = training_opts
        self.inference_opts = inference_opts
        self.nickname = None

    def setup(self, model):
        # call setup
        self.trainer.call_setup_hook(model)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        if self.half:
            model = model.half()

        # Wrap module
        self.train_model = IPUWrapperModule(model=model)

        # Create model for training which will run training.
        self.train_model = poptorch.trainingModel(
            model=self.train_model,
            options=self.training_opts,
            optimizer=self.trainer.optimizers[0]
        )

        self.validation_model = IPUWrapperModule(model=model, mode='validation')
        # Create model for training which will run validation.
        self.validation_model = poptorch.inferenceModel(
            model=self.validation_model,
            options=self.inference_opts,
        )

        self.test_model = IPUWrapperModule(model=model, mode='test')
        # Create model for training which will run testing.
        self.test_model = poptorch.inferenceModel(
            model=self.test_model,
            options=self.inference_opts,
        )

        self.trainer.model = self.train_model

    def train(self):
        model = self.trainer.model

        # set up training routine
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()
        return results

    def _step(self, model_step: Callable, args):
        args[0] = self.to_type(args[0])
        return model_step(*args[0])

    def training_step(self, args) -> None:
        args = self.to_type(args)
        return self._step(self.train_model, args)

    def validation_step(self, args):
        return self._step(self.validation_model, args)

    def test_step(self, args):
        return self._step(self.test_model, args)

    def to_type(self, batch):
        if self.mixed_precision or self.half:
            # Move tensors to half precision.
            return self.batch_to_device(batch, gpu_id)
        return batch

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        return tensor

    @property
    def require_distributed_sampler(self):
        return False

    @property
    def override_optimization(self):
        return True

    def on_reset_train_dataloader(self, dataloader: Union[DataLoader, Any]) -> Union[DataLoader, Any]:
        if isinstance(dataloader, DataLoader):
            dataloader = self._convert_to_poptorch_loader(
                dataloader=dataloader,
                opts=self.training_opts
            )
        return dataloader

    def on_reset_eval_dataloader(self, dataloader: Union[DataLoader, Any]) -> Union[DataLoader, Any]:
        if isinstance(dataloader, DataLoader):
            dataloader = self._convert_to_poptorch_loader(
                dataloader=dataloader,
                opts=self.inference_opts
            )
        return dataloader

    def _convert_to_poptorch_loader(self, dataloader, opts):
        skip_keys = ['dataset_kind']
        if dataloader.batch_size:
            # re-create batch sampler in new poptorch loader
            skip_keys += ['batch_sampler']

        dl_args = {
            k: v for k, v in dataloader.__dict__.items() if not k.startswith('_') and k not in skip_keys
        }
        dl_args["options"] = opts
        multiprocessing_context = dataloader.multiprocessing_context
        dataloader = poptorch.DataLoader(**dl_args)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader

    def get_reference_model(self, model) -> LightningModule:
        if not isinstance(model, LightningModule):
            return model._model.module
        return model

    @classmethod
    def add_argparse_args(cls, parser):
        parser = IPUOptsBuilder.add_argparse_args(parser)
        parser = IPUDebugOpts.add_argparse_args(parser)
        return parser

    @classmethod
    def parse_opts(cls, args):
        opts = IPUOptsBuilder(args)
        IPUDebugOpts.parse_environment_debug_opts(args)
        return opts.training_opts, opts.inference_opts


class IPUWrapperModule(torch.nn.Module):

    def __init__(self, model, mode='train'):
        super().__init__()
        self.module = model
        self.mode = mode
        self.have_warned_on_first_iteration = False

    def forward(self, *args, **kwargs):
        if not self.have_warned_on_first_iteration:
            rank_zero_info(
                "First iteration will take longer as we are tracing the step function "
                "and compiling the poplar graph for the IPUs."
            )
            self.have_warned_on_first_iteration = True

        if self.mode == 'train':
            output = self.module.training_step(args, **kwargs)
        elif self.mode == 'test':
            output = self.module.test_step(args, **kwargs)
        else:
            output = self.module.validation_step(args, **kwargs)
        return output
