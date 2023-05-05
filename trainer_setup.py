from pytorch_lightning.trainer import Trainer
import os, signal, logging
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config, get_obj_from_str
import ray
from ray.train.lightning import (
    LightningTrainer,
    LightningConfigBuilder,
    LightningCheckpoint,
)
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig


# merge_with_default_logger_config merges the default loggers
# configuration with that specified in the supplied config.
def merge_with_default_logger_config(logdir, logger_config):
    # default logger configs
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "tensorboard",
                "save_dir": logdir,
            }
        }
    }
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    return OmegaConf.merge(default_logger_cfg, logger_config)

# trainer_device_kwargs returns a dictionary of the kwargs to use
# for trainer that relate to the devices, accelerator and strategy to use.
# It also returns the number of gpus, if any, required.
def trainer_device_kwargs(config):  
    if not "gpus" in config:
        return dict(), 0
    cfg = dict()
    cfg["strategy"] = "ddp"
    cfg["accelerator"] = "gpu"
    cfg["devices"] = config["gpus"]
    print(f'Running on GPUs {cfg["devices"]}')
    return cfg, len(config["gpus"].strip(",").split(','))


def determine_learning_rate(batch_size, base_lr, num_gpus, accumulate_grad_batches, scale_lr):
    if scale_lr:
        lr = accumulate_grad_batches * num_gpus * batch_size * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                   lr, accumulate_grad_batches, num_gpus, batch_size, base_lr))
    else:
        lr = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {lr:.2e}")
    return lr

class LightningCallbacks:
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config, model_config, debug):

        # Note all other configs will be merged with this one.
        self.__default_callbacks = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": debug,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        self.__default_modelckpt = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
  
        self.__metrics_checkpoint = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }

        self.__model_checkpoint(model_config, lightning_config)
        self.__metrics(lightning_config)
        self.__merged_callbacks = self.__merge_callbacks(lightning_config)

    def __model_checkpoint(self, model_config, lightning_config):
        monitor = model_config.get("monitor", None)
        if monitor is not None:
            print(f"Monitoring {monitor} as checkpoint metric.")
            self.__default_modelckpt["params"]["monitor"] = monitor
            self.__default_modelckpt["params"]["save_top_k"] = 3

        model_checkpoint = lightning_config.get("modelcheckpoint", OmegaConf.create())

        model_checkpoint = OmegaConf.merge(self.__default_modelckpt, model_checkpoint)
        self.__default_callbacks.update({'checkpoint_callback': model_checkpoint})

        print(f"Merged modelckpt-cfg: \n{model_checkpoint}")

    def __metrics(self, lightning_config):
        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            self.__default_callbacks.update(self.__metrics_checkpoint)
        

    def __merge_callbacks(self, lightning_config):
        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
        callbacks_cfg = OmegaConf.merge(self.__default_callbacks, callbacks_cfg)
        return callbacks_cfg
    
    def configure_resume_from_checkpoint(self, resume_from_checkpoint: str):
        if 'ignore_keys_callback' in self.__merged_callbacks and resume_from_checkpoint != "":
            self.__merged_callbacks.ignore_keys_callback.params['ckpt_path'] = resume_from_checkpoint
        elif 'ignore_keys_callback' in self.__merged_callbacks:
            del self.__merged_callbacks['ignore_keys_callback']


    def kwargs_callbacks(self):    
       return [instantiate_from_config(self.__merged_callbacks[k]) for k in self.__merged_callbacks]

class TrainerBase:
     def __init__(self, trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, debug):
        self._trainer_kwargs = trainer_kwargs
        self._model_config = model_config
        self._learning_rate = learning_rate
        self._last_chkpt_path = last_chkpt_path
        self._debug = debug
        self._logdir = logdir
        self._trainer = None

     def save_debug_runs(self, resume):
        if self._debug and not resume and self._trainer.global_rank == 0:
            dst, name = os.path.split(self._logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(self._logdir, dst)

     def print_summary(self):
        if self._trainer.global_rank == 0:
            print(self._trainer.profiler.summary())

class UseLightningTrainer(TrainerBase):
    def __init__(self, trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, debug):
        super().__init__(trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, debug)
        
    def __create_trainer_and_model(self):
        model = instantiate_from_config(self._model_config)
        model.learning_rate = self._learning_rate
        trainer = Trainer(**self._trainer_kwargs)
        trainer.logdir = self._logdir
        return model, trainer

    def fit(self, data):
        model, trainer = self.__create_trainer_and_model()
        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                trainer.save_checkpoint(self._last_chkpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
    
        self._trainer = trainer
        try:
             self._trainer.fit(model, data)
        except Exception:
            if not self._debug:
                melk()
            raise        

    def test(self, data):
        model, trainer = self.__create_trainer_and_model()
        self._trainer = trainer
        self._trainer.test(model, data)


class UseRayLightningTrainer(TrainerBase):
    def __init__(self, ray_env, lightning_config, trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, debug):
        super().__init__(trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, debug)
        self.__ray_env = ray_env
        self.__lightning_config = lightning_config
        self.__ray_initialized = False
     
    def __ray_init(self):
        if self.__ray_initialized:
            return
        runtime_env = {
            "config": {"setup_timeout_seconds": 3600},
            "conda": self.__ray_env,
            "env_vars": {}
        }
        ray.init(runtime_env=runtime_env,
                 logging_level = logging.DEBUG,
                 log_to_driver=True)
        self.__ray_initialized = True

    def __create_trainer(self, data):
        model_class = get_obj_from_str(self._model_config["target"])
        model_params = self._model_config.get("params", dict())

        model_params["learning_rate"] = self._learning_rate
        lightning_config = (
            LightningConfigBuilder()
            .module(cls=model_class, **model_params)
            .trainer(**self._trainer_kwargs)
            .fit_params(datamodule=data)
            .build()
        )
 
        # read the scaling and run configs from the lightning block
        # of the config file.
        scaling_config = instantiate_from_config(self.__lightning_config.get("scaling_config", OmegaConf.create()))

        run_config = instantiate_from_config(self.__lightning_config.get("run_config", OmegaConf.create()))

        print(f"scaling config {scaling_config}")
        print(f"run config {run_config}")

        trainer = LightningTrainer(
            lightning_config=lightning_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )
        return trainer

    def fit(self, data):
        self._trainer = self.__create_trainer(data)
        self.__ray_init()
        self._trainer.fit()

    def test(self, data):
        self._trainer = self.__create_trainer(data)
        self.__ray_init()
        self._trainer.test(data)

