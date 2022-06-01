#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=sched_mit_hill
#SBATCH --time=0-04:00
#SBATCH -C centos7
#SBATCH -o /home/wesg/sindy_mio/results/output.out
#SBATCH -e /home/wesg/sindy_mio/results/error.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wesg@mit.edu
#SBATCH --array=1-50

python ../runtime_experiment.py -s hopf -t $SLURM_ARRAY_TASK_ID -n runtime_final -p 5


