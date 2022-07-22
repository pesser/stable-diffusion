import os
import torch
import fire


def prune_it(p):
    print(f"prunin' in path: {p}")
    size_initial = os.path.getsize(p)
    nsd = dict()
    sd = torch.load(p, map_location="cpu")
    print(sd.keys())
    for k in sd.keys():
        if k != "optimizer_states":
            nsd[k] = sd[k]
    else:
        print(f"removing optimizer states for path {p}")
    if "global_step" in sd:
        print(f"This is global step {sd['global_step']}.")
    fn = f"{os.path.splitext(p)[0]}-pruned.ckpt"
    print(f"saving pruned checkpoint at: {fn}")
    torch.save(nsd, fn)
    newsize = os.path.getsize(fn)
    print(f"New ckpt size: {newsize*1e-9:.2f} GB. "
          f"Saved {(size_initial - newsize)*1e-9:.2f} GB by removing optimizer states")


if __name__ == "__main__":
    fire.Fire(prune_it)
    print("done.")
