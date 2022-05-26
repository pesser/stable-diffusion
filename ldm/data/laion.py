import webdataset as wds
from PIL import Image
import io
import os
from tqdm import tqdm

if __name__ == "__main__":
    url = "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/000000.tar -"
    dataset = wds.WebDataset(url)
    example = next(iter(dataset))
    for k in example:
        print(k, type(example[k]))

    print(example["__key__"])
    for k in ["json", "txt"]:
        print(example[k].decode())

    image = Image.open(io.BytesIO(example["jpg"]))
    outdir = "tmp"
    os.makedirs(outdir, exist_ok=True)
    image.save(os.path.join(outdir, example["__key__"]+".png"))


    def load_example(example):
        return {
            "key": example["__key__"],
            "image": Image.open(io.BytesIO(example["jpg"])),
            "text": example["txt"].decode(),
        }


    for i, example in tqdm(enumerate(dataset)):
        ex = load_example(example)
        print(ex["image"].size, ex["text"])
        if i >= 100:
            break
