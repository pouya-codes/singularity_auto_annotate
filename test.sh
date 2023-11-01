#!/bin/bash
#SBATCH --job-name test
#SBATCH --cpus-per-task 6
#SBATCH --output /projects/ovcare/classification/cchen/ml/slurm/test.%j.out
#SBATCH --error  /projects/ovcare/classification/cchen/ml/slurm/test.%j.out
#SBATCH -w {w}
#SBATCH -p {p}
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --chdir /projects/ovcare/classification/cchen/ml/singularity_auto_annotate
#SBATCH --mem=70G

DLHOST04_SINGULARITY=/opt/singularity-3.4.0/bin
if [[ -d "$DLHOST04_SINGULARITY" ]]; then
    PATH="${PATH}:${DLHOST04_SINGULARITY}"
fi
if [[ -d /projects/ovcare/classification/cchen ]]; then
    cd /projects/ovcare/classification/cchen/ml/singularity_auto_annotate
    source /projects/ovcare/classification/cchen/{pyenv}
fi

mkdir -p auto_annotate/tests/outputs

pytest -s -vv auto_annotate/tests
