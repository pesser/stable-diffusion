import argparse
import os
import random
from dataclasses import dataclass, field
from io import BytesIO
import base64
import PIL

from generate import *

from eden.block import Block
from eden.hosting import host_block
from eden.datatypes import Image

eden_block = Block()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-workers', help='maximum number of workers to be run in parallel', required=False, default=1, type=int)
parser.add_argument('-p', '--port', help='localhost port', required=False, type=int, default=5656)
parser.add_argument('-rh', '--redis-host', help='redis host', required=False, type=str, default='localhost')
parser.add_argument('-rp', '--redis-port', help='redis port', required=False, type=int, default=6379)
parser.add_argument('-l', '--logfile', help='filename of log file', required=False, type=str, default=None)
args = parser.parse_args()


def b64str_to_PIL(data):
    data = data.replace('data:image/png;base64,', '')
    pil_img = PIL.Image.open(BytesIO(base64.b64decode(data)))
    return pil_img


@dataclass
class StableDiffusionSettings:
    input_image: PIL.Image = None
    mask_image: PIL.Image = None
    text_input: str = "a painting of a virus monster playing guitar"
    ddim_steps: int = 50
    plms: bool = False
    ddim_eta: float = 0.0
    n_iter: int = 1
    H: int = 256
    W: int = 256
    C: int = 4
    f: int = 8    
    n_samples: int = 8
    scale: float = 5.0
    dyn: float = None
    config: str = "logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml"
    ckpt: str = "logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt"
    seed: int = 42


def convert_samples_to_eden(samples, intermediate=False):
    results = {}
    if intermediate:
        results['intermediate_creation'] = Image(samples[0])
    else:
        results['creation'] = Image(samples[0])
        if len(samples) > 1:
            for s, sample in enumerate(samples):
                results[f'creation{s+1}'] = Image(sample)
    return results


my_args = {
    "mode": "generate",
    "input_image": "",
    "mask_image": "",
    "text_input": "Hello world", 
    "width": 512,
    "height": 512,
    "n_samples": 1,
    "n_iter": 1,
    "scale": 5.0,
    "ddim_steps": 50,
    "plms": True,
    "C": 16,
    "f": 8
}

@eden_block.run(args=my_args)
def run(config):
    
    mode = config["mode"]
    assert(mode in ["generate", "inpaint"], \
        f"Error: mode {mode} not recognized (generate or inpaint allowed)")

    settings = StableDiffusionSettings(
        text_input = config["text_input"],
        ddim_steps = config["ddim_steps"],
        scale = config["scale"],
        plms = config["plms"],
        n_samples = config["n_samples"],
        n_iter = config["n_iter"],
        ckpt = "f16-33k+12k-hr_pruned.ckpt",
        config = "configs/stable-diffusion/txt2img-multinode-clip-encoder-f16-768-laion-hr-inference.yaml",
        C = config['C'],
        f = config['f'],
        W = config["width"] - (config["width"] % 128),
        H = config["height"] - (config["height"] % 128)
    )

    def callback(intermediate_samples, i):
        config.progress.update(1 / settings.ddim_steps)
        if intermediate_samples:
            intermediate_results = convert_samples_to_eden(intermediate_samples, intermediate=True)
            eden_block.write_results(output=intermediate_results, token=config.token)

    if config["mode"] == "generate":
        final_samples = run_diffusion(settings, callback=callback, update_image_every=10)

    elif config["mode"] == "inpaint":
        input_image = b64str_to_PIL(config["input_image"])
        mask_image = b64str_to_PIL(config["mask_image"])
        output_image = run_inpainting(settings, input_image, mask_image, callback=callback, update_image_every=10)
        final_samples = [output_image]        
    
    results = convert_samples_to_eden(final_samples)

    return results



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
