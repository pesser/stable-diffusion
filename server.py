import argparse
import os
import random
import string

from eden.block import Block
from eden.hosting import host_block

eden_block = Block()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-workers', help='maximum number of workers to be run in parallel', required=False, default=1, type=int)
parser.add_argument('-p', '--port', help='localhost port', required=False, type=int, default=5656)
parser.add_argument('-rh', '--redis-host', help='redis host', required=False, type=str, default='localhost')
parser.add_argument('-rp', '--redis-port', help='redis port', required=False, type=int, default=6379)
parser.add_argument('-l', '--logfile', help='filename of log file', required=False, type=str, default=None)
args = parser.parse_args()


from generate import *

from dataclasses import dataclass, field

@dataclass
class StableDiffusionSettings:
    prompt: str = "hello world"
    outdir: str = "test_dir"
    skip_grid: bool = False
    skip_save: bool = False
    ddim_steps: int = 50
    plms: bool = True
    ddim_eta: float = 0.0
    n_iter: int = 1
    H: int = 256
    W: int = 256
    C: int = 4
    f: int = 8
    n_samples: int = 8
    n_rows: int = 0
    scale: float = 5.0
    config: str = "logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml"
    ckpt: str = "myModel.ckpt"
    seed: int = 42


my_args = {
    "prompt": "Hello world",    
}

@eden_block.run(args=my_args)
def run_stable_diffusion(config):
    
    prompt = config["prompt"]
    
    
    opt = StableDiffusionSettings(
        prompt = prompt,
        outdir = 'test_dir',
        skip_grid = False,
        skip_save = False,
        ddim_steps = 50,
        plms = True,
        ddim_eta = 0.0,
        n_iter = 1,
        H = 256,
        W = 256,
        C = 4,
        f = 8,
        n_samples = 8,
        n_rows = 0,
        scale = 5.0,
        config = "logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml",
        ckpt = "myModel.ckpt",
        seed = 42
    )

    result = generate(opt)

    return {
        "completion": result
    }


host_block(
    block = eden_block,
    port = args.port,
    host = "0.0.0.0",
    max_num_workers = args.num_workers,
    redis_port = args.redis_port,
    redis_host = args.redis_host,
    logfile = args.logfile, 
    log_level = 'debug',
    requires_gpu = True
)
