import os
import torch
import fire


def printit(p):
    print(f"printin' in path: {p}")
    size_initial = os.path.getsize(p)
    nsd = dict()
    sd = torch.load(p, map_location="cpu")
    if "global_step" in sd:
        print(f"This is global step {sd['global_step']}.")
    if "model_ema.num_updates" in sd["state_dict"]:
        print(f"And we got {sd['state_dict']['model_ema.num_updates']} EMA updates.")


if __name__ == "__main__":
    fire.Fire(printit)
