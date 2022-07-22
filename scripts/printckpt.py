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


if __name__ == "__main__":
    fire.Fire(printit)
