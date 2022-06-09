import argparse, os, sys, glob
import numpy as np
from torch_fidelity import calculate_metrics
import yaml

from ldm.modules.evaluate.evaluate_perceptualsim import compute_perceptual_similarity_from_list


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        nargs="?",
        default="fidelity-evaluation",
    )
    parser.add_argument(
        "--reconstructions",
        type=str,
        help="path to reconstructed images"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        help="path to input images"
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        help="optional, for pre-computed fidelity statistics",
        nargs="?",
    )
    return parser


if __name__ == "__main__":

    command = " ".join(sys.argv)
    np.random.RandomState(42)

    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    outdir = os.path.join(opt.logdir, "metrics")
    print(outdir)

    inppath = opt.inputs
    recpath = opt.reconstructions

    results = dict()

    ##### fid
    fid_kwargs = {}
    cache_root = None
    if opt.cache_root and os.path.isdir(opt.cache_root):
        print(f'Using cached Inception Features saved under "{cache_root}"')
        fid_kwargs.update({
            'cache_root': cache_root,
            'input2_cache_name': 'input_data',
            'cache': True
        })

    metrics_dict = calculate_metrics(input1=recpath, input2=inppath,
                                     cuda=True, isc=True, fid=True, kid=True,
                                     verbose=True, **fid_kwargs)

    results["fidelity"] = metrics_dict
    print(f'Metrics from fidelity: \n {results["fidelity"]}')

    ##### sim
    print("Evaluating reconstruction similarity")
    reconstructions = sorted(glob.glob(os.path.join(recpath, "*.png")))
    print(f"num reconstructions found: {len(reconstructions)}")
    inputs = sorted(glob.glob(os.path.join(inppath, "*.png")))
    print(f"num inputs found: {len(inputs)}")

    results["image-sim"] = compute_perceptual_similarity_from_list(
        reconstructions, inputs, take_every_other=False)
    print(f'Results sim: {results["image-sim"]}')

    # info
    results["info"] = {
        "n_examples": len(reconstructions),
        "command": command,
    }

    # write out
    ipath, rpath = map(lambda x: os.path.splitext(x)[0].split(os.sep)[-1], (inppath, recpath))
    resultsfn = f"results_{ipath}-{rpath}.yaml"
    results_file = os.path.join(outdir, resultsfn)
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(results_file)
    print("\ndone.")
