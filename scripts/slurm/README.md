# Example

Resume f8 @ 512 on Laion-HR

```
sbatch scripts/slurm/resume_512/sbatch.sh
```

# Reuse

To reuse this as a template, copy `sbatch.sh` and `launcher.sh` somewhere. In
`sbatch.sh`, adjust the lines

```
#SBATCH --job-name=stable-diffusion-512cont
#SBATCH --nodes=24
```

and the path to your `launcher.sh` in the last line,

```
srun bash /fsx/stable-diffusion/stable-diffusion/scripts/slurm/resume_512/launcher.sh
```

In `launcher.sh`, adjust `CONFIG` and `EXTRA`. Maybe give it a test run with
debug flags uncommented and a reduced number of nodes.
