import tempfile
import numpy as np
import torch
from PIL import Image
from typing import List
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from einops import rearrange
import time
from cog import BasePredictor, Path, Input, BaseModel
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):

        config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        self.model = load_model_from_config(
            config, "models/ldm/text2img-large/model.ckpt"
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        self.plms_sampler = PLMSSampler(self.model)
        self.ddim_sampler = DDIMSampler(self.model)

    def predict(
        self,
        prompt: str = Input(
            default="a painting of a virus monster playing guitar",
            description="Prompt for generating image.",
        ),
        n_samples: int = Input(
            default=8,
            description="Number of samples to produce for each given prompt.",
        ),
        scale: float = Input(
            default=5.0,
            description="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)).",
        ),
        ddim_steps: int = Input(
            default=50,
            description="Number of ddim sampling steps.",
        ),
        ddim_eta: float = Input(
            default=0.0,
            description="ddim eta (eta=0.0 corresponds to deterministic sampling).",
        ),
        plms: bool = Input(
            default=False,
            description="Whether use plms sampling.",
        ),
        seed: int = Input(
            default=42,
            description="The seed (for reproducible sampling).",
        ),
    ) -> List[ModelOutput]:

        seed_everything(seed)
        batch_size = n_samples
        data = [batch_size * [prompt]]

        sampler = self.plms_sampler if plms else self.ddim_sampler
        start_code = None

        # use the default setting for the following params for the demo
        n_iter = 1  # sample this often
        C = 4  # latent channels
        f = 8  # downsampling factor, most often 8 or 16
        H = 256  # image height, in pixel space
        W = 256  # image width, in pixel space

        output = []

        with torch.no_grad():
            with self.model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if not scale == 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta,
                            dynamic_threshold=None,
                            x_T=start_code,
                        )

                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        for i, x_sample in enumerate(x_samples_ddim):
                            x_sample = 255.0 * rearrange(
                                x_sample.cpu().numpy(), "c h w -> h w c"
                            )
                            output_path = Path(tempfile.mkdtemp()) / f"output_{i}.png"
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                str(output_path))
                            output.append(ModelOutput(image=output_path))

                toc = time.time()

        print(
            f"Sampling took {toc - tic}s, i.e. produced {n_iter * n_samples / (toc - tic):.2f} samples/sec."
            f" \nEnjoy."
        )

        return output


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
