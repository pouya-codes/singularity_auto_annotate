#!/bin/bash
#SBATCH --job-name toreadme
#SBATCH --cpus-per-task 1
#SBATCH --output /projects/ovcare/classification/cchen/ml/slurm/toreadme.%j.out
#SBATCH --error  /projects/ovcare/classification/cchen/ml/slurm/toreadme.%j.out
#SBATCH -w {w}
#SBATCH -p {p}
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

source /projects/ovcare/classification/cchen/{pyenv}
cd /projects/ovcare/classification/cchen/ml/singularity_auto_annotate

echo """# Auto Annotate

**Caution: you need to check the error log constantly to monitor _CUDA out of memory_ error. You need to use a smaller number for _--num_patch_workers_ if you get this error.**

To build the singularity image do:

\`\`\`
singularity build --remote auto_annotate.sif Singularityfile.def
\`\`\`

In the SH file, you should bind the path to the slides if the slides in your slides directory specified by \`--slide_location\` is symlinked.

\`\`\`
singularity run \
    -B /projects/ovcare/classification/cchen \
    -B /projects/ovcare/WSI \
    auto_annotate.sif \
    from-experiment-manifest /path/to/experiment.yaml \
\`\`\`

## Usage

\`\`\`""" > README.md
python app.py -h >> README.md
echo """\`\`\`

\`\`\`""" >> README.md
python app.py from-experiment-manifest -h >> README.md
echo """\`\`\`

\`\`\`""" >> README.md
python app.py from-arguments -h >> README.md
echo """\`\`\`
""" >> README.md
