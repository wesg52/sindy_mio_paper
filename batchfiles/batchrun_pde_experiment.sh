#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=newnodes
#SBATCH --time=0-04:00
#SBATCH -C centos7
#SBATCH -o /home/wesg/sindy_mio/results/output.out
#SBATCH -e /home/wesg/sindy_mio/results/error.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wesg@mit.edu
#SBATCH --array=1-50

python -u ../pde_experiment.py -s ks -t $SLURM_ARRAY_TASK_ID -n pde_final
python -u ../pde_experiment.py -s redif -t $SLURM_ARRAY_TASK_ID -n pde_final
