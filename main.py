import argparse, os, sys, datetime, glob
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config, get_obj_from_str

from pytorch_lightning.callbacks import LearningRateMonitor
from models import SetupCallback, ImageLogger, CUDACallback, DataModuleFromConfig

import trainer_setup as setup

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--ray-environment",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="set to a conda environment to enable using ray",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    return parser

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        model_config = config.model

        lightning_config = config.pop("lightning", OmegaConf.create())

        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        device_kwargs, num_gpus = setup.trainer_device_kwargs(trainer_config)

        instantiate_from_config(model_config) # purely for validation
        model_class = get_obj_from_str(model_config["target"])
        model_params = model_config.get("params", dict())

        # setup logger
        logger_cfg = setup.merge_with_default_logger_config(
            logdir, lightning_config.get("logger", OmegaConf.create()))
        
        logger_kwargs = dict()
        logger_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # setup lightning callbacks
        callback_kwargs = dict()

        lightning_callbacks = setup.LightningCallbacks(
           opt.resume, now, logdir, ckptdir, cfgdir, config, lightning_config, model_config, opt.debug)

        lightning_callbacks.configure_resume_from_checkpoint(trainer_config.get('resume_from_checkpoint', '"'))

        callback_kwargs["callbacks"] = lightning_callbacks.kwargs_callbacks()
        if not "plugins" in callback_kwargs:
            callback_kwargs["plugins"] = list()

        trainer_kwargs = dict()

        if not lightning_config.get("find_unused_parameters", True):
            from pytorch_lightning.plugins import DDPPlugin
            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))

        trainer_kwargs.update(device_kwargs)
        trainer_kwargs.update(logger_kwargs)
        trainer_kwargs.update(callback_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            print("datasets not yet initialized.")

        # configure learning rate         
        accumulate_grad_batches = trainer_config.get('accumulate_grad_batches', 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")

        learning_rate = setup.determine_learning_rate(
            config.data.params.batch_size,
            model_config.base_learning_rate,
            accumulate_grad_batches,
            num_gpus,
            opt.scale_lr,
        )

        last_chkpt_path = os.path.join(ckptdir, "last.ckpt")

        if opt.ray_environment != "":
            trainer = setup.UseRayLightningTrainer(opt.ray_environment, trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, opt.debug)
        else:
            trainer = setup.UseLightningTrainer(trainer_kwargs, model_config, learning_rate, last_chkpt_path, logdir, opt.debug)

        # run
        done = False
        if opt.train:
            trainer.fit(data)
            done = True

        if not opt.no_test and not done:
            trainer.test(data)

    except RuntimeError as err:
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise

    finally:
        # move newly created debug project to debug_runs
        trainer.save_debug_runs(opt.resume)
        trainer.print_summary()
