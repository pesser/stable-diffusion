import argparse
import os
import random
import base64
import PIL
from io import BytesIO

from settings import StableDiffusionSettings
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
    "text_inputs": ["Hello world"], 
    "input_image": "",
    "mask_image": "",
    "width": 512,
    "height": 512,
    "n_samples": 1,
    "n_interpolate": 1,
    "n_iter": 1,
    "scale": 5.0,
    "ddim_steps": 50,
    "plms": True,
    "C": 4,
    "f": 8,
    "fixed_code": False
}

@eden_block.run(args=my_args)
def run(config):
    
    mode = config["mode"]
    assert(mode in ["generate", "inpaint", "interpolate"], \
        f"Error: mode {mode} not recognized (generate or inpaint allowed)")

    settings = StableDiffusionSettings(
        text_inputs = config["text_inputs"],
        ddim_steps = config["ddim_steps"],
        scale = config["scale"],
        plms = config["plms"],
        n_samples = config["n_samples"],
        n_iter = config["n_iter"],
        n_interpolate = config["n_interpolate"],
        ckpt = "v1pp-flatlined-hr.ckpt",
        config = "configs/stable-diffusion/v1_improvedaesthetics.yaml", 
        C = config['C'],
        f = config['f'],
        W = config["width"] - (config["width"] % 128),
        H = config["height"] - (config["height"] % 128),
        fixed_code = config["fixed_code"]
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

    elif config["mode"] == "interpolate":

        # update schema: text_inputs
        # put in bucket
        
        # make a video
        # intermediate frames / callback function
        # if boomerang, cache frames


        # update collage
        # update react-app
        # update discord-bots

        

        run_diffusion_interpolation(settings)



        # output_image = run_inpainting(settings, input_image, mask_image, callback=callback, update_image_every=10)
        # final_samples = [output_image]        

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
