import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch_ldm(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
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


def make_batch_sd(
        image,
        mask,
        txt,
        device):
    # image hwc in -1 1
    image = np.array(Image.open(image).convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "jpg": image.to(device=device),
            "txt": [txt],
            "mask": mask.to(device=device),
            "masked_image": masked_image.to(device=device),
            }
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta of ddim",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=6.0,
        help="scale of unconditional guidance",
    )
    parser.add_argument(
        "--worldsize",
        type=int,
        default=1,
        help="scale of unconditional guidance",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="scale of unconditional guidance",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/fsx/robin/stable-diffusion/stable-diffusion/logs/2022-08-01T08-52-14_v1-finetune-for-inpainting-laion-aesthetic-larger-masks-and-ucfg/checkpoints/last.ckpt",
        help="scale of unconditional guidance",
    )
    opt = parser.parse_args()

    assert opt.rank < opt.worldsize

    mstr = "mask000.png"
    masks = sorted(glob.glob(os.path.join(opt.indir, f"*_{mstr}")))
    images = [x.replace(f"_{mstr}", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    #config = "models/ldm/inpainting_big/config.yaml"
    config="/fsx/stable-diffusion/stable-diffusion/configs/stable-diffusion/inpainting/v1-finetune-for-inpainting-laion-iaesthe.yaml"
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    #ckpt="models/ldm/inpainting_big/last.ckpt"
    ckpt=opt.ckpt
    model.load_state_dict(torch.load(ckpt)["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    indices = [i for i in range(len(images)) if i % opt.worldsize == opt.rank]
    images = [images[i] for i in indices]
    masks = [masks[i] for i in indices]

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks), total=len(images)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                #batch = make_batch_ldm(image, mask, device=device)

                ##### unroll
                batch = make_batch_sd(image, mask, txt="photograph of a beautiful empty scene, highest quality settings",
                        device=device)

                c = model.cond_stage_model.encode(batch["txt"])

                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = (1, 4, 64, 64)
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond={"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(1, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = (model.channels, model.image_size, model.image_size)
                samples_cfg, intermediates = sampler.sample(
                        opt.steps,
                        1,
                        shape,
                        cond,
                        verbose=False,
                        eta=opt.eta,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc_full,
                )
                x_samples_ddim = model.decode_first_stage(samples_cfg)

                image = torch.clamp((batch["jpg"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
