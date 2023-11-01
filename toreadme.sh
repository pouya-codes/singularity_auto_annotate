#!/bin/bash
#SBATCH --job-name toreadme
#SBATCH --cpus-per-task 1
#SBATCH --output /projects/ovcare/classification/cchen/ml/slurm/toreadme.%j.out
#SBATCH --error  /projects/ovcare/classification/cchen/ml/slurm/toreadme.%j.out
#SBATCH -w dlhost03
#SBATCH -p gpu2080
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

source /projects/ovcare/classification/cchen/c90
cd /projects/ovcare/classification/cchen/ml/docker_auto_annotate

python app.py -h >> README.md
