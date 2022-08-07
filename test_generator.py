from settings import StableDiffusionSettings
from generate import *
import cv2
from PIL import Image


def test_generate():
    opt = StableDiffusionSettings(
        mode = "generate",
        text_input = ['A cyberpunk city'],
        ddim_steps = 50,
        plms = True,
        H = 512,
        W = 512)

    results = run_diffusion(opt)
    Image.fromarray(results[0]).save('results/example.png')


def test_interpolation():
    opt = StableDiffusionSettings(
        mode = "interpolate",
        text_input = [
            'Steampunk robot face',
            'Noir detective with cyclops eye'
        ],
        n_interpolate = 20,
        ddim_steps = 50,
        plms = True,
        H = 512,
        W = 512,
        seed = 13,
        fixed_code=True)

    results = run_diffusion_interpolation(opt)

    out = cv2.VideoWriter('results/interpolation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, (512, 512))
    for frame in results:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

    out.release()


test_generate()
test_interpolation()