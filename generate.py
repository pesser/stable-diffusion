import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = None
inpainting_model = None
config = None


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def make_batch(image, mask, device):
    image = np.array(image.convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


def run_inpainting(opt, input_image, mask_image, callback=None, update_image_every=1):

    global inpainting_model
    if inpainting_model is None:
        config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
        inpainting_model = instantiate_from_config(config.model)
        inpainting_model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"], strict=False)
        inpainting_model = inpainting_model.to(device)

    sampler = DDIMSampler(inpainting_model)

    def inner_callback(img, i):
        intermediate_samples = None
        if i % update_image_every != 0:
            intermediate_samples = []
            x_samples_ddim = inpainting_model.decode_first_stage(img)
            image = np.array(input_image.convert("RGB"))
            image = image.astype(np.float32)/255.0
            image = image[None].transpose(0,3,1,2)
            image = torch.from_numpy(image)
            mask = np.array(mask_image.convert("L"))
            mask = mask.astype(np.float32)/255.0
            mask = mask[None,None]
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)
            image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"]+1.0)/2.0, min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            inpainted = (1-mask)*image+mask*predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
            intermediate_samples.append(inpainted.astype(np.uint8))
        callback(intermediate_samples, i)

    with torch.no_grad():
        with inpainting_model.ema_scope():
            batch = make_batch(input_image, mask_image, device=device)

            # encode masked image and concat downsampled mask
            c = inpainting_model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                             img_callback=inner_callback if callback else None,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)
            x_samples_ddim = inpainting_model.decode_first_stage(samples_ddim)

            image = torch.clamp((batch["image"]+1.0)/2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                            min=0.0, max=1.0)

            inpainted = (1-mask)*image+mask*predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
            output_image = inpainted.astype(np.uint8)

            return output_image


def run_diffusion(opt, callback=None, update_image_every=1):
    
    global model
    if model is None:        
        config = OmegaConf.load(f"{opt.config}")
        model = load_model_from_config(config, f"{opt.ckpt}")
        model = model.to(device)
    
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    batch_size = opt.n_samples

    prompt = opt.text_input
    assert prompt is not None
    data = [batch_size * [prompt]]
        
    all_samples = list()

    def inner_callback(img, i):
        intermediate_samples = None
        if i % update_image_every != 0:
            intermediate_samples = []
            x_samples_ddim = model.decode_first_stage(img)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                intermediate_samples.append(x_sample.astype(np.uint8))
        callback(intermediate_samples, i)
    
    with torch.no_grad():
        with model.ema_scope():
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H//opt.f, opt.W//opt.f]

                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     img_callback=inner_callback if callback else None,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     dynamic_threshold=opt.dyn)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        all_samples.append(x_sample.astype(np.uint8))

    return all_samples
    