# sindy_mio
Code for Learning Sparse Nonlinear Dynamics via Mixed-Integer Optimization

Our algorithm is self-contained in `miosr.py` and adheres to the [PySINDy](https://github.com/dynamicslab/pysindy) optimizer [interface](https://pysindy.readthedocs.io/en/latest/api/pysindy.optimizers.html). 
However, it does require a [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) license (free for academic use).
The rest of our dependencies are standard, and included in `requirements.txt`.


## Contents
- `batchfiles/` directory containing batch and slurm files to run experiment trials in parallel
- `results/` the raw csvs generated from the experiments used to generate the figures
- `miosr.py` contains implementation of main optimization model
- `*_experiment.py`  contain the outer loop of the experiment (e.g., varying different parameters and saving results)
- `models.py` and `weak_models.py` contain functions for tuning all algorithms
- `systems.py` contain methods for generating synthetic data
- `make_plots.py` generate the paper figures


