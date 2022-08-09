import argparse
import os
import random
import base64
import hashlib
import PIL
import moviepy.editor as mpy
from io import BytesIO

from minio import Minio
from minio.error import S3Error

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

minio_url = os.environ['MINIO_URL']
minio_bucket_name = os.environ['MINIO_BUCKET_NAME']
minio_access_key = os.environ['MINIO_ACCESS_KEY']
minio_secret_key = os.environ['MINIO_SECRET_KEY']

minio_client = Minio(
    minio_url,
    access_key=minio_access_key,
    secret_key=minio_secret_key
)

def get_file_sha256(filepath): 
    sha256_hash = hashlib.sha256()
    with open(filepath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
        sha = sha256_hash.hexdigest()
    return sha

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
    "text_input": "Hello world", 
    "input_image": "",
    "mask_image": "",
    "width": 512,
    "height": 512,
    "n_samples": 1,
    "interpolation_texts": [],
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
        mode = config["mode"],
        text_input = config["text_input"],
        ddim_steps = config["ddim_steps"],
        scale = config["scale"],
        plms = config["plms"],
        n_samples = config["n_samples"],
        n_iter = config["n_iter"],
        interpolation_texts = config["interpolation_texts"],
        n_interpolate = config["n_interpolate"],
        ckpt = "v1pp-flatlined-hr.ckpt",
        config = "configs/stable-diffusion/v1_improvedaesthetics.yaml", 
        C = config['C'],
        f = config['f'],
        W = config["width"] - (config["width"] % 128),
        H = config["height"] - (config["height"] % 128),
        fixed_code = config["fixed_code"]
    )

    def progress_callback(intermediate_samples):
        config.progress.update(1 / settings.ddim_steps)
        if intermediate_samples:
            intermediate_results = convert_samples_to_eden(intermediate_samples, intermediate=True)
            eden_block.write_results(output=intermediate_results, token=config.token)

    def video_callback(sample):
        num_frames = settings.n_interpolate * (len(settings.interpolation_texts) - 1)
        config.progress.update(1 / num_frames)
        intermediate_results = {'intermediate_creation': Image(sample)}
        eden_block.write_results(output=intermediate_results, token=config.token)

    if config["mode"] == "generate":
        final_samples = run_diffusion(settings, callback=progress_callback, update_image_every=10)
        results = convert_samples_to_eden(final_samples)
        return results

    elif config["mode"] == "interpolate":
        frames = run_diffusion_interpolation(settings, callback=video_callback)
        results = {"creation": Image(frames[0])}
        output_file = 'results/interpolation_%d.mp4' % random.randint(1, 1e8)
        clip = mpy.ImageSequenceClip(frames, fps=8)
        clip.write_videofile(output_file)
        video_sha = get_file_sha256(output_file)
        if video_sha:
            minio_client.fput_object(minio_bucket_name, video_sha+'.mp4', output_file)
            results["video_sha"] = video_sha        
        if os.path.isfile(output_file):
            os.remove(output_file)
        return results
        
    elif config["mode"] == "inpaint":
        input_image = b64str_to_PIL(config["input_image"])
        mask_image = b64str_to_PIL(config["mask_image"])
        output_image = run_inpainting(settings, input_image, mask_image, callback=progress_callback, update_image_every=10)
        results = convert_samples_to_eden([output_image])
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
