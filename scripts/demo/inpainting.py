import streamlit as st
import torch
import cv2
import numpy as np
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas


torch.set_grad_enabled(False)


def sample(
        model,
        prompt,
        n_runs=3,
        n_samples=2,
        H=512,
        W=512,
        scale=10.0,
        ddim_steps=50,
        callback=None,
        image=None,
        mask=None,
        ):
    batch = np2batch(image=image, mask=mask, txt=prompt)

    self = model
    unconditional_guidance_scale = scale
    unconditional_guidance_label = [""]
    use_ddim = True
    ddim_eta = 0
    N = 1
    ema_scope = self.ema_scope

    z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)
    c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

    if unconditional_guidance_scale > 1.0:
        uc_cross = self.get_unconditional_conditioning(N, unconditional_guidance_label)
        uc_cat = c_cat
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        with ema_scope("Sampling with classifier-free guidance"):
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            samples = self.decode_first_stage(samples_cfg)
    else:
        raise ValueError()

    samples = torch2np(samples)
    return samples


def np2batch(
        image,
        mask,
        txt):
    print("###")
    print(image.shape)
    print(mask.shape)
    print("###")
    # image hwc in -1 1
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 1
    mask = torch.from_numpy(mask)[:,:,:1]
    masked_image = image * (mask < 0.5)

    batch = {
            "jpg": image[None],
            "txt": [txt],
            "mask": mask[None],
            "masked_image": masked_image[None],
            }
    return batch


def torch2np(x):
    x = ((x+1.0)*127.5).clamp(0, 255).to(dtype=torch.uint8)
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    return x


@st.cache(allow_output_mutation=True)
def init():
    state = dict()
    return state


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")               
    pl_sd = torch.load(ckpt, map_location="cpu")      
    global_step = pl_sd.get("global_step", "?")
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
    print(f"Loaded global step {global_step}")
    return model


def run(
        config="/fsx/stable-diffusion/stable-diffusion/configs/stable-diffusion/inpainting/v1-finetune-for-inpainting-laion-iaesthe.yaml",
        #ckpt="/fsx/robin/stable-diffusion/stable-diffusion/logs/2022-07-28T07-44-05_v1-finetune-for-inpainting-laion-aesthetic-larger-masks/checkpoints/last.ckpt",
        ckpt="/fsx/robin/stable-diffusion/stable-diffusion/logs/2022-08-01T08-52-14_v1-finetune-for-inpainting-laion-aesthetic-larger-masks-and-ucfg/checkpoints/last.ckpt",
        ):
    st.title("Stable Inpainting")
    state = init()

    if not "model" in state:
        config = OmegaConf.load(config)
        model = load_model_from_config(config, ckpt)
        state["model"] = model

    uploaded_file = st.file_uploader("Upload image to inpaint")
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        width, height = image.size
        smaller = min(width, height)
        crop = (                    
            (width-smaller)//2,                                                          
            (height-smaller)//2,
            (width-smaller)//2+smaller,
            (height-smaller)//2+smaller,
        )
        image = image.crop(crop)
        image = image.resize((512, 512))
        #st.write("Uploaded Image")                                                       
        #st.image(image)

        st.write("Draw a mask (and send it to streamlit, button lower left)")
        stroke_width = int(st.number_input("Stroke Width", value=50))      
        canvas_result = st_canvas(                                
            fill_color="rgba(255, 255, 255)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="rgb(0, 0, 0)",
            background_color="rgb(0, 0, 0)",
            background_image=image if image is not None else Image.fromarray(255*np.ones((512,512,3),
                                                                     dtype=np.uint8)),
            update_streamlit=False,
            height=image.size[1] if image is not None else 512,
            width=image.size[0] if image is not None else 512,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="canvas",
        )
        if canvas_result:
            mask = canvas_result.image_data
            mask = np.array(mask)[:,:,[3,3,3]]
            mask = mask > 127

            # visualize
            bdry = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), dtype=np.uint8))
            bdry = (bdry > 0) & ~mask

            masked_image = np.array(image)*(1-mask) + mask*0.3*np.array(image)
            masked_image[:,:,0][bdry[:,:,0]] = 255
            masked_image[:,:,1][bdry[:,:,1]] = 0
            masked_image[:,:,2][bdry[:,:,2]] = 0
            st.write("Masked Image")
            st.image(Image.fromarray(masked_image.astype(np.uint8)))

            prompt = st.text_input("Prompt")
            scale = float(st.number_input("Guidance", value=10.0))
            t_total = int(st.number_input("Diffusion steps", value=50))

            if st.button("Sample"):
                st.text("Sampling")
                batch_progress = st.progress(0)
                batch_total = 3
                t_progress = st.progress(0)
                result = st.empty()
                #canvas = make_canvas(2, 3)
                def callback(x, batch, t):
                    #result.text(f"{batch}, {t}")
                    batch_progress.progress(min(1.0, (batch+1)/batch_total))
                    t_progress.progress(min(1.0, (t+1)/t_total))
                    update_canvas(canvas, x, batch)
                    result.image(canvas)

                samples = sample(
                        state["model"],
                        prompt,
                        n_runs=3,
                        n_samples=2,
                        H=512,
                        W=512,
                        scale=scale,
                        ddim_steps=t_total,
                        callback=callback,
                        image=np.array(image),
                        mask=np.array(mask),
                        )
                st.text("Samples")
                st.image(samples[0])


if __name__ == "__main__":
    import fire
    fire.Fire(run)
