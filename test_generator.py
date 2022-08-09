from settings import StableDiffusionSettings
from generate import *
import cv2
from PIL import Image


def test_generate():
    opt = StableDiffusionSettings(
        mode = "generate",
        text_input = 'A cyberpunk city',
        ddim_steps = 50,
        plms = True,
        H = 512,
        W = 512)

    results = run_diffusion(opt)
    Image.fromarray(results[0]).save('results/example.png')


def test_interpolation():
    opt = StableDiffusionSettings(
        mode = "interpolate",
        text_input = 'Steampunk robot face',
        interpolation_texts = [
            'Steampunk robot face',
            'Noir detective with cyclops eye'
        ],
        n_interpolate = 5,
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


def grid_search():
    text_inputs = [
        "laughing and crying faces everywhere",
        "many cute aliens with large eyes",
        "a scene from the matrix",
        "dogs playing poker",
        "Buddha at a rave",
        "water lillies",
        "a bridge in Tuscany under moonlight",
        "a Spaceship flies through the universe",
        "Intergalactic warfare with space cruisers",
        "a desert landscape",
        "vaporwave landscape",
        "retrowave landscape",
        "the italian countryside",
        "smurfs hijacking a car",
        "snow-capped mountains",
        "electronics market in Shenzhen",
        "lost souls trapped in the pits of hell",
        "an explosion of pastel colors",
        "code on the computer screen",
        "hello world",
        "leprechauns twerking",
        "strange creatures wearing dresses",
        "The god of air as a mystic wandering the Himalayas",
        "Crypto Punk 1629 Female with Black Lips Pink Hair and Hat on Purple Background",
        "Last supper with robots",
        "cute robots",
        "cyberpunk mall with electronics",
        "brain computer interface",
        "A flying taxi",
        "Right wing militia protesting climate refugees",
        "Cyborg riding electric unicycle in Tokyo at midnight",
        "A city in the year 2250"
    ]

    for seed in [13, 50, 100]:
        for t, text_input in enumerate(text_inputs):
            for d in [10, 20, 40, 60]:
                for plms in [True, False]:

                    opt = StableDiffusionSettings(
                        mode = "generate",
                        text_input = text_input,
                        ddim_steps = d,
                        plms = plms,
                        scale = 12,
                        seed = seed,
                        H = 512,
                        W = 512)

                    results = run_diffusion(opt)
                    Image.fromarray(results[0]).save('results/%s_%d_%d_%d.png' % (text_input, seed, d, 1 if plms else 0))



# test_generate()
# test_interpolation()
