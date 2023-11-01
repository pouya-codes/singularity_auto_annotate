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
