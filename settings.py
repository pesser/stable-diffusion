from dataclasses import dataclass, field
from typing import List
from PIL import Image


@dataclass
class StableDiffusionSettings:
    config: str = "configs/stable-diffusion/v1_improvedaesthetics.yaml"
    ckpt: str = "v1pp-flatlined-hr.ckpt"    

    mode: str = "generate"
    
    text_input: List = field(default_factory=lambda: [
        "a painting of a virus monster playing guitar"
    ])
    input_image: Image = None
    mask_image: Image = None
    
    seed: int = 42
    fixed_code: bool = False

    ddim_steps: int = 50
    plms: bool = False
    ddim_eta: float = 0.0
    C: int = 4
    f: int = 8    
    scale: float = 5.0
    dyn: float = None

    H: int = 256
    W: int = 256
    n_iter: int = 1
    n_samples: int = 1
    n_interpolate: int = 1
