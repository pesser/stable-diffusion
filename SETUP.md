to setup:

```
conda env create -f environment.yaml
conda init zsh
conda activate ldm
pip install torchmetrics==0.6.0
```

to run:

```
conda activate ldm
python3 server.py --num-workers 1 --port 5656 --redis-host eden-dev-gene-redis --redis-port 6379
```