import os
import re
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as torch_distrib
import torch.multiprocessing as mp

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.cluster_environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.distributed.dist import LightningDistributed
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities.cloud_io import atomic_save, load as pl_load
from pytorch_lightning.utilities.distributed import find_free_network_port, rank_zero_only
from pytorch_lightning.utilities.distributed import sync_ddp_if_available, rank_zero_warn
from pytorch_lightning.utilities.seed import seed_everything

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:

    class ReduceOp:
        SUM = None


class DDPSpawnPlugin(ParallelPlugin):

    distributed_backend = "ddp_spawn"

    def __init__(
        self,
        parallel_devices,
        num_nodes=1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm=False,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment)
        self.num_nodes = num_nodes
        self.sync_batchnorm = sync_batchnorm
        self._ddp_kwargs = kwargs
        self.dist = LightningDistributed()
        self.num_processes = len(parallel_devices)
        self.node_rank = 0
        self.mp_queue = None

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def lightning_module(self):
        # the model may not be wrapped with DistributedDataParallel if calling this too early
        return getattr(self._model, "module", self._model)

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    def setup(self, model):
        self._model = model

        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(find_free_network_port()))

        # pass in a state q
        smp = mp.get_context("spawn")
        self.mp_queue = smp.SimpleQueue()

    def set_world_ranks(self, process_idx):
        self.local_rank = process_idx
        self.node_rank = self.cluster_environment.node_rank()
        self.global_rank = self.node_rank * self.num_processes + self.local_rank
        self.world_size = self.num_nodes * self.num_processes

    def start_training(self, trainer):
        mp.spawn(self.new_process, nprocs=self.num_processes, args=(trainer,))
        # reset optimizers, since main process is never used for training and thus does not have a valid optim state
        trainer.optimizers = []

    def start_testing(self, trainer):
        mp.spawn(self.new_process, nprocs=self.num_processes, args=(trainer,))

    def new_process(self, process_idx, trainer):
        # TODO: check if needed
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))

        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.init_ddp_connection(self.global_rank, self.world_size)

        # TODO: we moved it to the trainer.fit after calling pre_training
        #   ... need to double check that it is the correct place
        # self.trainer.call_setup_hook(self.model)

        # on world_size=0 let everyone know training is starting
        if self.is_global_zero and not torch.distributed.is_initialized():
            log.info("-" * 100)
            log.info(f"distributed_backend={self.distributed_backend}")
            log.info(f"All DDP processes registered. Starting ddp with {self.world_size} processes")
            log.info("-" * 100)

        # set the ranks and devices
        self.dist.rank = self.global_rank
        self.dist.device = self.root_device

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # move the model to the correct device
        self.model_to_device()

        self.configure_ddp()

        self.barrier()

        if trainer.testing:
            results = trainer.run_test()
        else:
            results = trainer.train()

        # persist info in ddp_spawn
        self.transfer_distrib_spawn_state_on_fit_end(results)

    def post_training(self):
        # restore main state with best weights
        best_path = self.mp_queue.get()
        last_path = self.mp_queue.get()
        self._results = self.mp_queue.get()

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(best_path, last_path)

    def configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        self.model = LightningDistributedDataParallel(
            self.model,
            device_ids=self.determine_ddp_device_ids(),
            **self._ddp_kwargs,
        )

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        # TODO: this code is duplicated in DDP and DDPSpawn, make this a function
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())
        torch_backend = "nccl" if self.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch_distrib.init_process_group(torch_backend, rank=global_rank, world_size=world_size)

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def transfer_distrib_spawn_state_on_fit_end(self, results):
        # TODO: is there a better way than accessing callback through model -> trainer -> callback?
        best_model_path = self.lightning_module.trainer.checkpoint_callback.best_model_path

        if self.global_rank == 0 and self.mp_queue is not None:
            rank_zero_warn("cleaning up ddp environment...")

            # save the last weights
            last_path = None
            # TODO: is there a better way than accessing trainer through model -> trainer?
            if not self.lightning_module.trainer.testing and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub(".ckpt", ".tmp_end.ckpt", best_model_path)
                atomic_save(self.lightning_module.state_dict(), last_path)

            # todo, pass complete checkpoint as state dictionary
            self.mp_queue.put(best_model_path)
            self.mp_queue.put(last_path)
            self.mp_queue.put(results)

    def __recover_child_process_weights(self, best_path, last_path):
        # TODO: is there a better way than accessing callback through model -> trainer -> callback?
        # transfer back the best path to the trainer
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also best score

        # load last weights
        if last_path is not None and not self.lightning_module.trainer.testing:
            ckpt = pl_load(last_path, map_location=lambda storage, loc: storage)
            self.lightning_module.load_state_dict(ckpt)

    def barrier(self, *args, **kwargs):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        return self.dist.broadcast(obj)

    def model_to_device(self):
        if self.root_device.type == "cuda":
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if isinstance(output, torch.Tensor):
            output = sync_ddp_if_available(output, group, reduce_op)
        return output
