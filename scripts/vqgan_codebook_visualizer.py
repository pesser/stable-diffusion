import argparse, os, sys, glob
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from einops import rearrange
from torchvision.utils import make_grid


rescale = lambda x: (x + 1.) / 2.


def bchw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(0,2,3,1))

def chw_to_st(x):
    return rescale(x.detach().cpu().numpy().transpose(1,2,0))


def custom_get_input(batch, key):
    inputs = batch[key].permute(0, 3, 1, 2)
    return inputs


def vq_no_codebook_forward(model, x):
    h = model.encoder(x)
    h = model.quant_conv(h)
    h = model.post_quant_conv(h)
    xrec = model.decoder(h)
    return xrec


def save_img(x, fname):
    I = (x.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(I).save(fname)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        nargs="?",
        default="",
        help="path to dataset config"
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


@st.cache(allow_output_mutation=True)
def load_model_and_dset(config, ckpt, gpu, eval_mode, delete_dataset_params=False):
    # get data
    if delete_dataset_params:
        st.info("Deleting dataset parameters.")
        del config["data"]["params"]["train"]["params"]
        del config["data"]["params"]["validation"]["params"]

    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


@torch.no_grad()
def get_image_embeddings(model, dset, used_codebook, used_indices):
    import plotly.graph_objects as go
    batch_size = st.number_input("Batch size for embedding visualization", min_value=1, value=4)
    start_index = st.number_input("Start index", value=0,
                                          min_value=0,
                                          max_value=len(dset) - batch_size)
    if st.sidebar.button("Sample Batch"):
        indices = np.random.choice(len(dset), batch_size)
    else:
        indices = list(range(start_index, start_index + batch_size))

    st.write(f"Indices: {indices}")
    batch = default_collate([dset[i] for i in indices])
    x = model.get_input(batch, "image")
    x = x.to(model.device)

    # get reconstruction from non-quantized and quantized, compare
    z_pre_quant = model.encode_to_prequant(x)
    z_quant, emb_loss, info = model.quantize(z_pre_quant)
    indices = info[2].detach().cpu().numpy()
    #indices = rearrange(indices, '(b d) -> b d', b=batch_size)
    unique_indices = np.unique(indices)
    st.write(f"Unique indices in batch: {unique_indices.shape[0]}")

    x1 = used_codebook[:, 0].cpu().numpy()
    x2 = used_codebook[:, 1].cpu().numpy()
    x3 = used_codebook[:, 2].cpu().numpy()

    zp1 = rearrange(z_pre_quant, 'b c h w -> (b h w) c')[:, 0].cpu().numpy()
    zp2 = rearrange(z_pre_quant, 'b c h w -> (b h w) c')[:, 1].cpu().numpy()
    zp3 = rearrange(z_pre_quant, 'b c h w -> (b h w) c')[:, 2].cpu().numpy()

    zq1 = rearrange(z_quant, 'b c h w -> (b h w) c')[:, 0].cpu().numpy()
    zq2 = rearrange(z_quant, 'b c h w -> (b h w) c')[:, 1].cpu().numpy()
    zq3 = rearrange(z_quant, 'b c h w -> (b h w) c')[:, 2].cpu().numpy()

    fig = go.Figure(data=[go.Scatter3d(x=x1, y=x2, z=x3, mode='markers', marker=dict(size=1.4, line=dict(width=1.,
                                                                                                         color="Blue")),
                                       name="All Used Codebook Entries"),

                          ])
    trace2 = go.Scatter3d(x=zp1, y=zp2, z=zp3, mode='markers', marker=dict(size=1., line=dict(width=1., color=indices)),
                          name="Pre-Quant Codes")
    trace3 = go.Scatter3d(x=zq1, y=zq2, z=zq3, mode='markers',
                          marker=dict(size=2., line=dict(width=10., color=indices)), name="Quantized Codes")

    fig.add_trace(trace2)
    fig.add_trace(trace3)

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
    )

    x_rec_no_quant = model.decode(z_pre_quant)
    x_rec_quant = model.decode(z_quant)
    delta_x = x_rec_no_quant - x_rec_quant

    st.text("Fitting Gaussian...")
    h, w = z_quant.shape[2], z_quant.shape[3]
    from sklearn.mixture import GaussianMixture
    gaussian = GaussianMixture(n_components=1)
    gaussian.fit(rearrange(z_pre_quant, 'b c h w -> (b h w) c').cpu().numpy())
    samples, _ = gaussian.sample(n_samples=batch_size*h*w)
    samples = rearrange(samples, '(b h w) c -> b h w c', b=batch_size, h=h, w=w, c=3)
    samples = rearrange(samples, 'b h w c -> b c h w')
    samples = torch.tensor(samples).to(z_quant)
    samples, _, _ = model.quantize(samples)
    x_sample = model.decode(samples)

    all_img = torch.stack([x, x_rec_quant, x_rec_no_quant, delta_x, x_sample])   # 5 b 3 H W

    all_img = rearrange(all_img, 'n b c h w -> b n c h w')
    all_img = rearrange(all_img, 'b n c h w -> (b n) c h w')
    grid = make_grid(all_img, nrow=5)

    st.write("** Input | Rec. (w/ quant) | Rec. (no quant) | Delta(quant, no_quant) **")
    st.image(chw_to_st(grid), clamp=True, output_format="PNG")

    st.write(fig)
    # 2d projections
    import matplotlib.pyplot as plt
    pairs = [(1, 0), (2, 0), (2, 1)]
    fig2, ax = plt.subplots(1, 3, figsize=(21, 7))
    for d in range(3):
        d1, d2 = pairs[d]
        #ax[d].scatter(used_codebook[:, d1].cpu().numpy(),
        #              used_codebook[:, d2].cpu().numpy(),
        #              label="All Used Codebook Entries", s=10.0, c=used_indices)
        ax[d].scatter(rearrange(z_quant, 'b c h w -> (b h w) c')[:, d1].cpu().numpy(),
                      rearrange(z_quant, 'b c h w -> (b h w) c')[:, d2].cpu().numpy(),
                      label="Quantized Codes", alpha=0.9, s=8.0, c=indices)
        ax[d].scatter(rearrange(z_pre_quant, 'b c h w -> (b h w) c')[:, d1].cpu().numpy(),
                      rearrange(z_pre_quant, 'b c h w -> (b h w) c')[:, d2].cpu().numpy(),
                      label="Pre-Quant Codes", alpha=0.5, s=1.0, c=indices)
        ax[d].set_title(f"dim {d2} vs dim {d1}")
        ax[d].legend()
    st.write(fig2)

    plt.close()

    # from scipy.spatial import Voronoi, voronoi_plot_2d
    # fig3 = plt.figure(figsize=(10,10))
    # points = rearrange(z_pre_quant, 'b c h w -> (b h w) c')[:, :2].cpu().numpy()
    # vor = Voronoi(points)
    # # plot
    # voronoi_plot_2d(vor)
    # # colorize
    # for region in vor.regions:
    #     if not -1 in region:
    #         polygon = [vor.vertices[i] for i in region]
    #         plt.fill(*zip(*polygon))
    # plt.savefig("voronoi_test.png")
    # st.write(fig3)


@torch.no_grad()
def get_used_indices(model, dset, batch_size=20):
    dloader = torch.utils.data.DataLoader(dset, shuffle=True, batch_size=batch_size, drop_last=False)
    data = list()
    info = st.empty()
    for i, batch in enumerate(dloader):
        x = model.get_input(batch, "image")
        x = x.to(model.device)
        zq, _, zi = model.encode(x)
        indices = zi[2]
        indices = indices.reshape(zq.shape[0], -1).detach().cpu().numpy()
        data.append(indices)

        unique = np.unique(data)
        info.text(f"iteration {i} [{batch_size*i}/{len(dset)}]: unique indices found so far: {unique.size}")

    unique = np.unique(data)
    #np.save(outpath, unique)
    st.write(f"end of data: found **{unique.size} unique indices.**")
    print(f"end of data: found {unique.size} unique indices.")
    return unique


def visualize3d(codebook, used_indices):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    codebook = codebook.cpu().numpy()
    selected_codebook = codebook[used_indices, :]
    z_dim = codebook.shape[1]
    assert z_dim == 3

    pairs = [(1,0), (2,0), (2,1)]
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for d in range(3):
        d1, d2 = pairs[d]
        ax[d].scatter(selected_codebook[:, d1], selected_codebook[:, d2])
        ax[d].set_title(f"dim {d2} vs dim {d1}")
    st.write(fig)

    # # plot 3D
    # fig = plt.figure(1)
    # ax = Axes3D(fig)
    # ax.scatter(codebook[:, 0], codebook[:, 1], codebook[:, 2], s=10., alpha=0.8, label="all entries")
    # ax.scatter(selected_codebook[:, 0], selected_codebook[:, 1], selected_codebook[:, 2], s=3., alpha=1.0, label="used entries")
    # plt.legend()
    # #st.write(fig)
    # st.pyplot(fig)

    # plot histogram of vector norms
    fig = plt.figure(2, figsize=(6,5))
    norms = np.linalg.norm(selected_codebook, axis=1)
    plt.hist(norms, bins=100, edgecolor="black", lw=1.1)
    plt.title("Distribution of norms of used codebook entries")
    st.write(fig)

    # plot 3D with plotly
    import pandas as pd
    import plotly.graph_objects as go
    x = selected_codebook[:, 0]
    y = selected_codebook[:, 1]
    z = selected_codebook[:, 2]
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2., line=dict(width=1.,
                                                                                                     color="Blue"))
                                       )
                          ]
                    )

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
    )
    st.write(fig)


@torch.no_grad()
def get_fixed_points(model, dset):
    n_iter = st.number_input("Number of Iterations for FP-Analysis", min_value=1, value=25)
    batch_size = st.number_input("Batch size for fixed-point visualization", min_value=1, value=4)
    start_index = st.number_input("Start index", value=0, min_value=0, max_value=len(dset) - batch_size)
    clip_decoded = st.checkbox("Clip decoded image", False)
    quantize_decoded = st.checkbox("Quantize decoded image (e.g. map back to uint8)", False)
    factor = st.sidebar.number_input("image size", value=1., min_value=0.1)
    if st.sidebar.button("Sample Batch"):
        indices = np.random.choice(len(dset), batch_size)
    else:
        indices = list(range(start_index, start_index + batch_size))

    st.write(f"Indices: {indices}")
    batch = default_collate([dset[i] for i in indices])
    x = model.get_input(batch, "image")
    x = x.to(model.device)

    progress = st.empty()
    progress_cb = lambda k: progress.write(f"iteration {k}/{n_iter}")

    image_progress = st.empty()  # TODO

    input = x
    img_quant = x
    img_noquant = x
    delta_img = img_quant - img_noquant

    st.write("** Input | Rec. (w/ quant) | Rec. (no quant) | Delta(quant, no_quant) **")

    def display(input, img_quant, img_noquant, delta_img):
        all_img = torch.stack([input, img_quant, img_noquant, delta_img])  # 4 b 3 H W
        all_img = rearrange(all_img, 'n b c h w -> b n c h w')
        all_img = rearrange(all_img, 'b n c h w -> (b n) c h w')
        grid = make_grid(all_img, nrow=4)
        image_progress.image(chw_to_st(grid), clamp=True, output_format="PNG", width=int(factor*grid.shape[2]))

    display(input, img_quant, img_noquant, delta_img)
    for n in range(n_iter):
        # get reconstruction from non-quantized and quantized, compare via iteration

        # quantized_stream
        z_pre_quant = model.encode_to_prequant(img_quant)
        z_quant, emb_loss, info = model.quantize(z_pre_quant)

        # non_quantized stream
        z_noquant = model.encode_to_prequant(img_noquant)

        img_quant = model.decode(z_quant)
        img_noquant = model.decode(z_noquant)
        if clip_decoded:
            img_quant = torch.clamp(img_quant, -1., 1.)
            img_noquant = torch.clamp(img_noquant, -1., 1.)
        if quantize_decoded:
            device = img_quant.device
            img_quant = (2*torch.Tensor(((img_quant.cpu().numpy()+1.)*127.5).astype(np.uint8))/255. - 1.).to(device)
            img_noquant = (2*torch.Tensor(((img_noquant.cpu().numpy()+1.)*127.5).astype(np.uint8))/255. - 1.).to(device)
        delta_img = img_quant - img_noquant
        display(input, img_quant, img_noquant, delta_img)
        progress_cb(n + 1)


@torch.no_grad()
def get_fixed_points_kl_ae(model, dset):
    n_iter = st.number_input("Number of Iterations for FP-Analysis", min_value=1, value=25)
    batch_size = st.number_input("Batch size for fixed-point visualization", min_value=1, value=4)
    start_index = st.number_input("Start index", value=0, min_value=0, max_value=len(dset) - batch_size)
    clip_decoded = st.checkbox("Clip decoded image", False)
    quantize_decoded = st.checkbox("Quantize decoded image (e.g. map back to uint8)", False)
    sample_posterior = st.checkbox("Sample from encoder posterior", False)
    factor = st.sidebar.number_input("image size", value=1., min_value=0.1)
    if st.sidebar.button("Sample Batch"):
        indices = np.random.choice(len(dset), batch_size)
    else:
        indices = list(range(start_index, start_index + batch_size))

    st.write(f"Indices: {indices}")
    batch = default_collate([dset[i] for i in indices])
    x = model.get_input(batch, "image")
    x = x.to(model.device)

    progress = st.empty()
    progress_cb = lambda k: progress.write(f"iteration {k}/{n_iter}")

    st.write("** Input | Rec. (no quant) | Delta(input, iter_rec) **")
    image_progress = st.empty()

    input = x
    img_noquant = x
    delta_img = input - img_noquant

    def display(input, img_noquant, delta_img):
        all_img = torch.stack([input, img_noquant, delta_img])  # 3 b 3 H W
        all_img = rearrange(all_img, 'n b c h w -> b n c h w')
        all_img = rearrange(all_img, 'b n c h w -> (b n) c h w')
        grid = make_grid(all_img, nrow=3)
        image_progress.image(chw_to_st(grid), clamp=True, output_format="PNG", width=int(factor*grid.shape[2]))

    fig, ax = plt.subplots()

    distribution_progress = st.empty()
    def display_latent_distribution(latent_z, alpha=1., title=""):
        flatz = latent_z.reshape(-1).cpu().detach().numpy()
        #fig, ax = plt.subplots()
        ax.hist(flatz, bins=42, alpha=alpha, lw=.1, edgecolor="black")
        ax.set_title(title)
        distribution_progress.pyplot(fig)

    display(input, img_noquant, delta_img)
    for n in range(n_iter):
        # get reconstructions

        posterior = model.encode(img_noquant)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if n==0:
            flatz_init = z.reshape(-1).cpu().detach().numpy()
            std_init = flatz_init.std()
            max_init, min_init = flatz_init.max(), flatz_init.min()

        display_latent_distribution(z, alpha=np.sqrt(1/(n+1)),
                                    title=f"initial z: std/min/max: {std_init:.2f}/{min_init:.2f}/{max_init:.2f}")

        img_noquant = model.decode(z)
        if clip_decoded:
            img_noquant = torch.clamp(img_noquant, -1., 1.)
        if quantize_decoded:
            img_noquant = (2*torch.Tensor(((img_noquant.cpu().numpy()+1.)*127.5).astype(np.uint8))/255. - 1.).to(model.device)
        delta_img = img_noquant - input
        display(input, img_noquant, delta_img)
        progress_cb(n + 1)



if __name__ == "__main__":
    from ldm.models.autoencoder import AutoencoderKL
    # VISUALIZE USED AND ALL INDICES of VQ-Model. VISUALIZE FIXED POINTS OF KL MODEL
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if opt.dataset_config:
        dcfg = OmegaConf.load(opt.dataset_config)
        print("Replacing data config with:")
        print(dcfg.pretty())
        dcfg = OmegaConf.to_container(dcfg)
        config["data"] = dcfg["data"]

    st.sidebar.text(ckpt)
    gs = st.sidebar.empty()
    gs.text(f"Global step: ?")
    st.sidebar.text("Options")
    gpu = st.sidebar.checkbox("GPU", value=True)
    eval_mode = st.sidebar.checkbox("Eval Mode", value=True)
    show_config = st.sidebar.checkbox("Show Config", value=False)
    if show_config:
        st.info("Checkpoint: {}".format(ckpt))
        st.json(OmegaConf.to_container(config))

    delelete_dataset_parameters = st.sidebar.checkbox("Delete parameters of dataset.")
    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode,
                                                    delete_dataset_params=delelete_dataset_parameters)
    gs.text(f"Global step: {global_step}")

    split = st.sidebar.radio("Split", sorted(dsets.datasets.keys())[::-1])
    dset = dsets.datasets[split]

    batch_size = st.sidebar.number_input("Batch size", min_value=1, value=20)
    num_batches = st.sidebar.number_input("Number of batches", min_value=1, value=5)
    data_size = batch_size*num_batches
    dset = torch.utils.data.Subset(dset, np.random.choice(np.arange(len(dset)), size=(data_size,), replace=False))

    if not isinstance(model, AutoencoderKL):
        # VQ MODEL
        codebook = model.quantize.embedding.weight.data
        st.write(f"VQ-Model has codebook of dimensionality **{codebook.shape[0]} x {codebook.shape[1]} (num_entries x z_dim)**")
        st.write(f"Evaluating codebook-usage on **{config['data']['params'][split]['target']}**")
        st.write("**Select ONE of the following options**")

        if st.checkbox("Show Codebook Statistics", False):
            used_indices = get_used_indices(model, dset, batch_size=batch_size)
            visualize3d(codebook, used_indices)
        if st.checkbox("Show Batch Encodings", False):
            used_indices = get_used_indices(model, dset, batch_size=batch_size)
            get_image_embeddings(model, dset, codebook[used_indices, :], used_indices)
        if st.checkbox("Show Fixed Points of Data", False):
            get_fixed_points(model, dset)

    else:
        st.info("Detected a KL model")
        get_fixed_points_kl_ae(model, dset)