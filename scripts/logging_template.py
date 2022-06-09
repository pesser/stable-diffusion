import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
import streamlit as st
from streamlit import caching
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from tqdm import tqdm
import datetime

from ldm.util import instantiate_from_config
from main import DataModuleFromConfig, ImageLogger, SingleImageLogger

rescale = lambda x: (x + 1.) / 2.

class DummyLogger:
    pass

def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))


def run(model, dsets, callbacks, logdir, split="train",
        batch_size=8, start_index=0, sample_batch=False, nowname="", use_full_data=False):
    logdir = os.path.join(logdir, nowname)
    os.makedirs(logdir, exist_ok=True)

    dset = dsets.datasets[split]
    print(f"Dataset size: {len(dset)}")
    dloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size, drop_last=False, shuffle=False)
    if not use_full_data:
        if sample_batch:
            indices = np.random.choice(len(dset), batch_size)
        else:
            indices = list(range(start_index, start_index+batch_size))
        print(f"Data indices: {list(indices)}")
        example = default_collate([dset[i] for i in indices])
        for cb in callbacks:
            if isinstance(cb, ImageLogger):
                print(f"logging with {cb.__class__.__name__}")
                cb.log_img(model, example, 0, split=split, save_dir=logdir)
    else:
        for batch in tqdm(dloader, desc="Data"):
            for cb in callbacks:
                if isinstance(cb, SingleImageLogger):
                    cb.log_img(model, batch, 0, split=split, save_dir=logdir)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
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
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "-n",
        "--n_iter",
        type=int,
        default=1,
        help="how many times to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="how many examples in the batch",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="evaluate on this split",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="eval_logs",
        help="where to save the logs",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default="state_dict",
        choices=["state_dict", "model_ema", "model"],
        help="where to access the model weights",
    )
    parser.add_argument(
        "--full_data",
        action='store_true',
        help="evaluate on full dataset",
    )
    parser.add_argument(
        "--ignore_callbacks",
        action='store_true',
        help="ignores all callbacks in the config and only uses main.SingleImageLogger",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    print("loading model from state-dict...")
    if sd is not None:
        m, u = model.load_state_dict(sd)
        if len(m) > 0: print(f"missing keys: \n {m}")
        if len(u) > 0: print(f"unexpected keys: \n {u}")
        print("loaded model.")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def get_callbacks(lightning_config, ignore_callbacks=False):
    callbacks_cfg = lightning_config.callbacks
    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    print(f"found and instantiated the following callback(s):")
    for cb in callbacks:
        print(f"  > {cb.__class__.__name__}")
    print()
    if len(callbacks) == 0 or ignore_callbacks:
        del callbacks
        callbacks = list()
        print("No callbacks found. Falling back to SingleImageLogger as a default")
        try:
            callbacks.append(SingleImageLogger(1, max_images=opt.batch_size, log_always=True,
                                           log_images_kwargs=lightning_config.callbacks.image_logger.params.log_images_kwargs))
        except:
            print("No log_images_kwargs specified. Using SingleImageLogger with default values in log_images().")
            callbacks.append(SingleImageLogger(1, max_images=opt.batch_size, log_always=True))
    return callbacks


@st.cache(allow_output_mutation=True)
def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        try:
            global_step = pl_sd["global_step"]
        except:
            global_step = 0
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   #pl_sd["state_dict"],
                                   pl_sd[opt.state_key],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


def exists(x):
    return x is not None


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    if not st._is_running_with_streamlit:
        print("Not running with streamlit. Redefining st functions...")
        st.info = print
        st.write = print

    seed_everything(42)
    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    assert opt.resume
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))

    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        try:
            idx = len(paths)-paths[::-1].index("logs")+1
        except ValueError:
            idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), opt.resume
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
    opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    lightning_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-lightning.yaml")))
    lightning_configs = [OmegaConf.load(lcfg) for lcfg in lightning_configs]
    lightning_config = OmegaConf.merge(*lightning_configs, cli)

    print(f"ckpt-path: {ckpt}")

    print(config)
    print(lightning_config)

    gpu = True
    eval_mode = True

    callbacks = get_callbacks(lightning_config.lightning, ignore_callbacks=opt.ignore_callbacks)

    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")

    logdir = os.path.join(logdir, opt.logdir, f"{global_step:09}")
    print(f"logging to {logdir}")
    os.makedirs(logdir, exist_ok=True)

    # go
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    for n in range(opt.n_iter):
        nowname = now + "_iteration-" + f"{n:03}"
        run(model, dsets, callbacks, logdir=logdir, batch_size=opt.batch_size, nowname=nowname,
            split=opt.split, use_full_data=opt.full_data)
