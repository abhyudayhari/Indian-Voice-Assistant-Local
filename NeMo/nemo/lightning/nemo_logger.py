import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import lightning_fabric as fl
import pytorch_lightning as pl
from fiddle._src.experimental import serialization
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.utils import logging
from nemo.utils.app_state import AppState


@dataclass
class NeMoLogger(IOMixin):
    """Logger for NeMo runs.

    Args:
        name (str): Name of the experiment.
        dir (Optional[str]): Directory to save logs.
        explicit_log_dir (Optional[str]): Explicit log directory.
        version (Optional[str]): Version of the experiment.
        use_datetime_version (bool): Whether to use datetime as version.
        log_local_rank_0_only (bool): Log only on local rank 0.
        log_global_rank_0_only (bool): Log only on global rank 0.
        files_to_copy (Optional[List[str]]): List of files to copy to log directory.
        update_logger_directory (bool): Whether to update logger directory.
        ckpt (Optional[ModelCheckpoint]): Model checkpoint callback.
    """

    name: str = "default"
    dir: Optional[str] = None
    explicit_log_dir: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: bool = True
    log_local_rank_0_only: bool = False
    log_global_rank_0_only: bool = False
    files_to_copy: Optional[List[str]] = None
    update_logger_directory: bool = True
    ckpt: Optional[ModelCheckpoint] = None
    tensorboard: Optional[TensorBoardLogger] = None
    wandb: Optional[WandbLogger] = None
    extra_loggers: List[Logger] = field(default_factory=list)

    def __post_init__(self):
        if self.log_local_rank_0_only is True and self.log_global_rank_0_only is True:
            raise ValueError(
                f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
            )

    def setup(self, trainer: Union[pl.Trainer, fl.Fabric], resume_if_exists: bool = False, task_config=None):
        """Setup the logger for the experiment.

        Args:
            trainer (Union[pl.Trainer, fl.Fabric]): Trainer or Fabric instance.
            resume_if_exists (bool): Whether to resume if log directory exists.

        Returns:
            AppState: The application state with updated log directory and other settings.
        """
        from nemo.constants import NEMO_ENV_VARNAME_VERSION
        from nemo.utils.exp_manager import check_explicit_log_dir
        from nemo.utils.get_rank import is_global_rank_zero

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = trainer.node_rank * trainer.world_size + self.local_rank
        logging.rank = self.global_rank

        if self.explicit_log_dir and isinstance(trainer, pl.Trainer):  # If explicit log_dir was passed, short circuit
            return check_explicit_log_dir(trainer, self.explicit_log_dir, self.dir, self.name, self.version)

        # Default dir to ./nemo_experiments if None was passed
        _dir = self.dir
        if self.dir is None:
            _dir = str(Path.cwd() / 'nemo_experiments')

        if not self.name:
            self.name = "default"

        version = self.version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)
        if is_global_rank_zero():
            if self.use_datetime_version:
                version = time.strftime('%Y-%m-%d_%H-%M-%S')
        if resume_if_exists:
            logging.warning(
                "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
            )
            version = None
        if version:
            if is_global_rank_zero():
                os.environ[NEMO_ENV_VARNAME_VERSION] = version

        log_dir = Path(_dir) / Path(str(self.name)) / Path("" if version is None else str(version))
        # update app_state with log_dir, exp_dir, etc
        app_state = AppState()
        app_state.log_dir = log_dir
        app_state.exp_dir = _dir
        app_state.name = self.name
        app_state.version = version
        app_state.cmd_args = sys.argv

        os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
        logging.info(f'Experiments will be logged at {log_dir}')

        if task_config and is_global_rank_zero():
            self._handle_task_config(task_config, log_dir)

        if isinstance(trainer, pl.Trainer):
            self._setup_trainer_loggers(trainer, _dir, version)
            self._setup_trainer_model_checkpoint(trainer, log_dir=log_dir, ckpt=self.ckpt)

        self._setup_files_to_move(log_dir, app_state)
        self._setup_file_logging(log_dir)

        return app_state

    def _setup_trainer_loggers(self, trainer, dir, version):
        loggers = [self.tensorboard, self.wandb, *self.extra_loggers]
        loggers = [logger for logger in loggers if logger is not None]

        if self.update_logger_directory and self.wandb:
            self.wandb._save_dir = dir
            self.wandb._wandb_init["dir"] = dir
            self.wandb._wandb_init["name"] = self.name
            self.wandb._name = self.name

        if loggers:
            if trainer.logger is not None and not self.tensorboard:
                loggers = [trainer.logger] + loggers
            trainer._logger_connector.configure_logger(loggers)

        if trainer.logger is not None:
            trainer.logger._version = version or ""
            if self.update_logger_directory:
                logging.warning(
                    f'"update_logger_directory" is True. Overwriting logger "save_dir" to {dir} and "name" to {self.name}'
                )
                trainer.logger._root_dir = dir
                trainer.logger._name = self.name

    def _setup_trainer_model_checkpoint(self, trainer, log_dir, ckpt=None):
        if ckpt:
            _overwrite_i = None
            for i, callback in enumerate(trainer.callbacks):
                if isinstance(callback, PTLModelCheckpoint):
                    logging.warning(
                        "The Trainer already contains a ModelCheckpoint callback. " "This will be overwritten."
                    )
                    _overwrite_i = i
                    break
            if _overwrite_i is not None:
                trainer.callbacks[_overwrite_i] = ckpt
            else:
                trainer.callbacks.append(ckpt)

            if ckpt.monitor and "val" in ckpt.monitor:
                if (
                    trainer.max_epochs is not None
                    and trainer.max_epochs != -1
                    and trainer.max_epochs < trainer.check_val_every_n_epoch
                ):
                    logging.error(
                        "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                        f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch({trainer.check_val_every_n_epoch}"
                        f"). It is very likely this run will fail with ModelCheckpoint(monitor='{ckpt.monitor}') not found "
                        "in the returned metrics. Please ensure that validation is run within trainer.max_epochs."
                    )
                elif trainer.max_steps is not None and trainer.max_steps != -1:
                    logging.warning(
                        "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                        f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                        f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
                    )

        for callback in trainer.callbacks:
            if isinstance(callback, PTLModelCheckpoint):
                if callback.dirpath is None:
                    callback.dirpath = Path(log_dir / "checkpoints")
                if callback.filename is None:
                    callback.filename = f'{self.name}--{{{callback.monitor}:.4f}}-{{epoch}}'
                ModelCheckpoint.CHECKPOINT_NAME_LAST = callback.filename + '-last'

    def _handle_task_config(self, task_config, log_dir):
        task_config.save_config_img(log_dir / "task.png")
        task_json = serialization.dump_json(task_config)
        with open(log_dir / "task.json", "w") as f:
            f.write(task_json)

    def _setup_file_logging(self, log_dir):
        """Set up file logging based on rank settings."""
        from nemo.constants import NEMO_ENV_VARNAME_TESTING
        from nemo.utils.env_var_parsing import get_envbool
        from nemo.utils.mcore_logger import add_handlers_to_mcore_logger

        # This is set if the env var NEMO_TESTING is set to True.
        nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)
        log_file = log_dir / f'nemo_log_globalrank-{self.global_rank}_localrank-{self.local_rank}.txt'

        if self.log_local_rank_0_only and not nemo_testing and self.local_rank == 0:
            logging.add_file_handler(log_file)
        elif self.log_global_rank_0_only and not nemo_testing and self.global_rank == 0:
            logging.add_file_handler(log_file)
        elif not (self.log_local_rank_0_only or self.log_global_rank_0_only):
            logging.add_file_handler(log_file)

        add_handlers_to_mcore_logger()

    def _setup_files_to_move(self, log_dir, app_state):
        files_to_move = []
        if Path(log_dir).exists():
            for child in Path(log_dir).iterdir():
                if child.is_file():
                    files_to_move.append(child)

        app_state.files_to_move = files_to_move
        app_state.files_to_copy = self.files_to_copy
