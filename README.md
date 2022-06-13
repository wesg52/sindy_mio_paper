# sindy_mio
Code for reproducing [Learning Sparse Nonlinear Dynamics via Mixed-Integer Optimization](https://arxiv.org/abs/2206.00176).

Our algorithm is self-contained in `miosr.py` and adheres to the [PySINDy](https://github.com/dynamicslab/pysindy) optimizer [interface](https://pysindy.readthedocs.io/en/latest/api/pysindy.optimizers.html). 
However, it does require a [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) license (free for academic use).
The rest of our dependencies are standard, and included in `requirements.txt`.

## Example Usage
Using our algorithm is essentially exactly the same as using any of the other optimizers within
the PySINDy library, except the basic hyperparameter that requires tuning is the exact sparsity level
(either for each dimension or across all dimensions) as opposed to a threshold.

```
import numpy as np
import pysindy as ps
from miosr import MIOSR
from scipy.integrate import solve_ivp
from pysindy.utils import linear_3D

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

dt = .01
t_train = np.arange(0, 50, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [2, 0, 1]
x_train = solve_ivp(linear_3D, t_train_span,
                    x0_train, t_eval=t_train, **integrator_keywords).y.T

poly_order = 5
target_sparsity = 5

model = ps.SINDy(
    # Could also set group_sparsity=(2, 2, 1)
    optimizer=MIOSR(target_sparsity=target_sparsity),
    feature_library=ps.PolynomialLibrary(degree=poly_order)
)
model.fit(x_train, t=dt)
model.print()
```


## Contents
- `batchfiles/` directory containing batch and slurm files to run experiment trials in parallel
- `results/` the raw csvs generated from the experiments used to generate the figures
- `miosr.py` contains implementation of main optimization model
- `*_experiment.py`  contain the outer loop of the experiment (e.g., varying different parameters and saving results)
- `models.py` and `weak_models.py` contain functions for tuning all algorithms
- `systems.py` contain methods for generating synthetic data
- `make_plots.py` generate the paper figures


## Cite
If you found this repo helpful, please consider citing our paper
```
@article{bertsimas2022learning,
  title={Learning Sparse Nonlinear Dynamics via Mixed-Integer Optimization},
  author={Bertsimas, Dimitris and Gurnee, Wes},
  journal={arXiv preprint arXiv:2206.00176},
  year={2022}
}
```


