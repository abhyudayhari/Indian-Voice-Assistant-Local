# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import signal
from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer

from nemo.utils import logging


class PreemptionCallback(Callback):
    """
    PreemptionCallback checks for preemption during training at the end of every step.
    Upon preemption, it signals the trainer to stop gracefully.

    Args:
        sig (int, optional): The signal to listen for. Defaults to signal.SIGTERM.

    Example:
        >>> from nemo.lightning.pytorch.callbacks import PreemptionCallback
        >>> callback = PreemptionCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self, sig: Optional[int] = None):
        self.sig = sig if sig is not None else signal.SIGTERM
        self._interrupted = False
        self._handler_context = None
        self._preemption_supported = None

    def on_train_start(self, trainer: Trainer, pl_module) -> None:
        if self.preemption_supported:
            self._handler_context = self._preemption_handler()
            self._handler_context.__enter__()

    def on_train_batch_start(self, trainer: Trainer, pl_module, batch, batch_idx: int) -> None:
        if not self.preemption_supported:
            self._preemption_supported = self._check_preemption_support()
            if self.preemption_supported:
                self._handler_context = self._preemption_handler()
                self._handler_context.__enter__()

    def on_train_end(self, trainer: Trainer, pl_module) -> None:
        if self._handler_context:
            self._handler_context.__exit__(None, None, None)

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self.interrupted:
            logging.info("Preemption detected, signaling trainer to stop")
            trainer.should_stop = True

    def on_exception(self, trainer: Trainer, pl_module, exception: BaseException) -> None:
        if isinstance(exception, PreemptionException):
            logging.info("Handling PreemptionException")
            trainer.should_stop = True

    @contextlib.contextmanager
    def _preemption_handler(self):
        if not self.preemption_supported:
            logging.warning("Preemption requires torch distributed to be initialized, preemption may be disabled")
            yield
            return

        original_handler = signal.getsignal(self.sig)

        def master_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful stop")
            self._interrupted = True
            raise PreemptionException("Preemption signal received")

        def ignoring_handler(signum, frame):
            logging.debug(f"Received signal {signum} on non-master rank, ignoring")

        try:
            private_rank = torch.distributed.get_rank()
            signal.signal(self.sig, master_handler if private_rank == 0 else ignoring_handler)
            yield
        finally:
            signal.signal(self.sig, original_handler)

    @property
    def preemption_supported(self) -> bool:
        if self._preemption_supported is None:
            self._preemption_supported = self._check_preemption_support()
        return self._preemption_supported

    def _check_preemption_support(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @property
    def interrupted(self) -> bool:
        if not self.preemption_supported:
            return False
        interrupted = torch.tensor(self._interrupted, device=torch.cuda.current_device(), dtype=torch.int32)
        torch.distributed.broadcast(interrupted, 0)
        return bool(interrupted.item())


class PreemptionException(Exception):
    """Custom exception for preemption events."""
