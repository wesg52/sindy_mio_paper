#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-04:00
#SBATCH -C centos7
#SBATCH -o /home/wesg/clusterrepo/myoutputfile.out
#SBATCH -e /home/wesg/clusterrepo/myerrorfile.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wesg@mit.edu
#SBATCH --array=1-10

python ../model_selection_experiment.py -s lorenz -t $SLURM_ARRAY_TASK_ID -n model_selection
python ../model_selection_experiment.py -s lorenz96 -t $SLURM_ARRAY_TASK_ID -n model_selection
python ../model_selection_experiment.py -s vanderpol -t $SLURM_ARRAY_TASK_ID -n model_selection
python ../model_selection_experiment.py -s duffing -t $SLURM_ARRAY_TASK_ID -n model_selection
python ../model_selection_experiment.py -s rossler -t $SLURM_ARRAY_TASK_ID -n model_selection


