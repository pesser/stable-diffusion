import os
import glob
import subprocess
import time
import fire

import numpy as np
from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
import contextlib


def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    gs = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    model.cuda()
    model.eval()
    return model, gs


def read_prompts(path):
    with open(path, "r") as f:
        prompts = f.read().splitlines()
    return prompts


def split_in_batches(iterator, n):
    out = []
    for elem in iterator:
        out.append(elem)
        if len(out) == n:
            yield out
            out = []
    if len(out) > 0:
        yield out


class Sampler(object):
    def __init__(self, out_dir, ckpt_path, cfg_path, prompts_path, shape, seed=42):
        self.out_dir = out_dir
        self.ckpt_path = ckpt_path
        self.cfg_path = cfg_path
        self.prompts_path = prompts_path
        self.seed = seed

        self.batch_size = 1
        self.scale = 10
        self.shape = shape
        self.n_steps = 100
        self.nrow = 8


    @torch.inference_mode()
    def sample(self, model, prompts, ema=True):
        seed = self.seed
        batch_size = self.batch_size
        scale = self.scale
        n_steps = self.n_steps

        shape = self.shape

        print("Sampling model.")
        print("ckpt_path", self.ckpt_path)
        print("cfg_path", self.cfg_path)
        print("prompts_path", self.prompts_path)
        print("out_dir", self.out_dir)
        print("seed", self.seed)
        print("batch_size", batch_size)
        print("scale", scale)
        print("n_steps", n_steps)
        print("shape", shape)

        prompts = list(split_in_batches(prompts, batch_size))

        sampler = PLMSSampler(model)
        all_samples = list()

        ctxt = model.ema_scope if ema else contextlib.nullcontext

        with ctxt():
            for prompts_batch in tqdm(prompts, desc="prompts"):
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                c = model.get_learned_conditioning(prompts_batch)

                seed_everything(seed)

                samples_latent, _ = sampler.sample(
                    S=n_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    dynamic_threshold=None,
                )

                samples = model.decode_first_stage(samples_latent)
                samples = torch.clamp((samples+1.0)/2.0, min=0.0, max=1.0)

                all_samples.append(samples)

        all_samples = torch.cat(all_samples, 0)
        return all_samples


    @torch.inference_mode()
    def __call__(self):
        config = OmegaConf.load(self.cfg_path)
        model, global_step = load_model_from_config(config, self.ckpt_path)
        print(f"Restored model at global step {global_step}.")

        prompts = read_prompts(self.prompts_path)

        all_samples = self.sample(model, prompts, ema=True)
        self.save_as_grid("grid_with_wings", all_samples, global_step)
        all_samples = self.sample(model, prompts, ema=False)
        self.save_as_grid("grid_without_wings", all_samples, global_step)


    def save_as_grid(self, name, grid, global_step):
        grid = make_grid(grid, nrow=self.nrow)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

        os.makedirs(self.out_dir, exist_ok=True)
        filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
            name,
            global_step,
            0,
            0,
        )
        grid_path = os.path.join(self.out_dir, filename)
        Image.fromarray(grid.astype(np.uint8)).save(grid_path)
        print(f"---> {grid_path}")


class Checker(object):
    def __init__(self, ckpt_path, callback, wait_for_file=5, interval=60):
        self._cached_stamp = 0
        self.filename = ckpt_path
        self.callback = callback
        self.interval = interval
        self.wait_for_file = wait_for_file

    def check(self):
        while True:
            if not os.path.exists(self.filename):
                print(f"Could not find {self.filename}. Waiting.")
                time.sleep(self.interval)
                continue

            stamp = os.stat(self.filename).st_mtime
            if stamp != self._cached_stamp:
                while True:
                    # try to wait until checkpoint is fully written
                    previous_stamp = stamp
                    time.sleep(self.wait_for_file)
                    stamp = os.stat(self.filename).st_mtime
                    if stamp != previous_stamp:
                        print(f"File is still changing. Waiting {self.wait_for_file} seconds.")
                    else:
                        break

                self._cached_stamp = stamp
                # file has changed, so do something...
                print(f"{self.__class__.__name__}: Detected a new file at "
                      f"{self.filename}, calling back.")
                self.callback()

            else:
                time.sleep(self.interval)


def run(prompts_path="scripts/prompts/prompts-with-wings.txt",
        watch_log_dir=None, out_dir=None, ckpt_path=None, cfg_path=None,
        H=256,
        W=None,
        C=4,
        F=8,
        wait_for_file=5,
        interval=60):

    if out_dir is None:
        assert watch_log_dir is not None
        out_dir = os.path.join(watch_log_dir, "images/checker")

    if ckpt_path is None:
        assert watch_log_dir is not None
        ckpt_path = os.path.join(watch_log_dir, "checkpoints/last.ckpt")

    if cfg_path is None:
        assert watch_log_dir is not None
        configs = glob.glob(os.path.join(watch_log_dir, "configs/*-project.yaml"))
        cfg_path = sorted(configs)[-1]

    if W is None:
        assert H is not None
        W = H
    if H is None:
        assert W is not None
        H = W
    shape = [C, H//F, W//F]
    sampler = Sampler(out_dir, ckpt_path, cfg_path, prompts_path, shape=shape)

    checker = Checker(ckpt_path, sampler, wait_for_file=wait_for_file, interval=interval)
    checker.check()


if __name__ == "__main__":
    fire.Fire(run)
