to setup:

```
conda env create -f environment.yaml
conda init zsh
```

to run:

```
conda activate ldm
python3 server.py -n "1" -p "5656" -rh eden-dev-gene-redis -rp "6379"
```
