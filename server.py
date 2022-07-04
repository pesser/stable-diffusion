import argparse
import os
import random
from dataclasses import dataclass, field

from generate import *

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


@dataclass
class StableDiffusionSettings:
    prompt: str = "a painting of a virus monster playing guitar"
    outdir: str = "outputs/txt2img-samples"
    skip_grid: bool = False
    skip_save: bool = False
    ddim_steps: int = 50
    plms: bool = False
    ddim_eta: float = 0.0
    n_iter: int = 1
    H: int = 256
    W: int = 256
    C: int = 4
    f: int = 8    
    n_samples: int = 8
    n_rows: int = 0
    scale: float = 5.0
    dyn: float = None
    from_file: str = None
    config: str = "logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml"
    ckpt: str = "logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt"
    seed: int = 42


my_args = {
    "prompt": "Hello world",    
}
@eden_block.run(args=my_args)
def run(config):
    
    prompt = config["prompt"]
    
    opt1 = StableDiffusionSettings(
        prompt=prompt,
        outdir="results",
        scale=5.0,
        plms=True,
        n_samples=4,
        n_iter=1,
        ckpt="f16-33k+12k-hr_pruned.ckpt",
        config="configs/stable-diffusion/txt2img-multinode-clip-encoder-f16-768-laion-hr-inference.yaml",
        C=16,
        f=16,
        H=512,
        W=512
    )

    result = run_diffusion(opt1)

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
