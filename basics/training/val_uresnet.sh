#!/bin/bash 

#SBATCH --account=ml
#SBATCH --partition=ml

#SBATCH --job-name=val_uresnet
#SBATCH --output=batch_outputs/output-val_uresnet.txt 
#SBATCH --error=batch_outputs/output-val_uresnet.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=72:00:00 
#SBATCH --gpus a100:1

singularity exec --bind /sdf/group/neutrino/drielsma/,/sdf/group/neutrino/ldomine/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 /sdf/group/neutrino/drielsma/lartpc_mlreco3d/bin/run.py /sdf/group/neutrino/drielsma/train/icarus/localized/uresnet_val.cfg"
