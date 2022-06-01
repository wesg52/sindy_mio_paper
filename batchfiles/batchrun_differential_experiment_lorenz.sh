#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-10:00
#SBATCH -C centos7
#SBATCH -o /home/wesg/sindy_mio/results/output.out
#SBATCH -e /home/wesg/sindy_mio/results/error.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wesg@mit.edu
#SBATCH --array=1-50

python ../differential_experiment.py -s lorenz -t $SLURM_ARRAY_TASK_ID -o 5 -n final_draft_dif
