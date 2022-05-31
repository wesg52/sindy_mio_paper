import argparse
import datetime
import pandas as pd
from pysindy.differentiation import SmoothedFiniteDifference
import os
import time
from miosr import *
from models import *
from systems import *


def runtime_experiment(
        system,
        seed=0,
        durations=(0.5, 1, 2, 4, 8),
        noise_fraction=0.001,
        dt=0.001,
        threshold=0.05,
        alpha=0.01,
        n_ensemble_models=50,
        group_sparsity=(2, 2),
        poly_order=5,
        smooth_window_length=9,
        run_warmstart=False
):
    feature_library = ps.PolynomialLibrary(degree=poly_order)
    differentiation_method = SmoothedFiniteDifference(
        smoother_kws={'window_length': smooth_window_length}
    )
    results = {}
    for duration in durations:
        model_times = []

        x0 = system.sample_initial_conditions(n=1, seed=seed)[0]
        t, x = system.simulate(duration, dt, x0=x0)
        library_dim = feature_library.fit_transform(x).shape[1]
        rmse = mean_squared_error(x, np.zeros(x.shape), squared=False)
        np.random.seed(seed)
        x_train = x + np.random.normal(0, rmse * noise_fraction, x.shape)

        preprocessing_start = time.time()
        x_dot = differentiation_method._differentiate(x_train, t)
        x_train = feature_library.fit_transform(x_train)
        preprocessing_t = time.time() - preprocessing_start
        model_times.append(('Preprocessing', preprocessing_t))
        id_lib = ps.IdentityLibrary()

        ensemble_optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
        model = ps.SINDy(
            optimizer=ensemble_optimizer,
            feature_library=id_lib
        )
        ensemble_start_t = time.time()
        model.fit(x_train, x_dot=x_dot, t=dt, ensemble=True, n_models=n_ensemble_models, quiet=True, unbias=False)
        ensemble_time = time.time() - ensemble_start_t
        model_times.append(('E-STLSQ', ensemble_time))

        stlsq_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
            feature_library=id_lib,
            differentiation_method=differentiation_method
        )
        stlsq_start_t = time.time()
        stlsq_model.fit(x_train, x_dot=x_dot, t=dt, quiet=True, unbias=False)
        stlsq_time = time.time() - stlsq_start_t
        model_times.append(('STLSQ', stlsq_time))

        ssr_model = ps.SINDy(
            optimizer=ps.SSR(max_iter=library_dim - 1, alpha=alpha),
            feature_library=id_lib,
            differentiation_method=differentiation_method
        )
        ssr_start_t = time.time()
        ssr_model.fit(x_train, x_dot=x_dot, t=dt, quiet=True, unbias=False)
        ssr_time = time.time() - ssr_start_t
        model_times.append(('SSR', ssr_time))
        ssr_coefs = np.array(ssr_model.optimizer.history_)
        mio_warmstart = ssr_coefs[np.array(group_sparsity), np.arange(x_dot.shape[1]), :]

        sr3_model = ps.SINDy(
            optimizer=ps.SR3(threshold=threshold, nu=0.1, max_iter=10000),
            feature_library=id_lib,
            differentiation_method=differentiation_method
        )
        sr3_start_t = time.time()
        sr3_model.fit(x_train, x_dot=x_dot, t=dt, quiet=True, unbias=False)
        sr3_time = time.time() - sr3_start_t
        model_times.append(('SR3', sr3_time))

        mio_model = ps.SINDy(
            optimizer=MIOSR(group_sparsity=group_sparsity, alpha=alpha),
            feature_library=id_lib,
            differentiation_method=differentiation_method
        )
        mio_start_t = time.time()
        mio_model.fit(x_train, x_dot=x_dot, t=dt, quiet=True, unbias=False)
        mio_time = time.time() - mio_start_t
        model_times.append(('MIOSR', mio_time))
        model_times.append(('MIOSR-Opt', sum(mio_model.optimizer.solve_times)))
        model_times.append(('MIOSR-Build', sum(mio_model.optimizer.build_times)))

        if run_warmstart:
            warm_mio_model = ps.SINDy(
                optimizer=MIOSR(group_sparsity=group_sparsity, alpha=alpha, initial_guess=mio_warmstart),
                feature_library=id_lib,
                differentiation_method=differentiation_method
            )
            mio_warm_start_t = time.time()
            warm_mio_model.fit(x_train, x_dot=x_dot, t=dt, quiet=True, unbias=False)
            mio_warm_time = time.time() - mio_warm_start_t
            model_times.append(('Warm-MIOSR', mio_warm_time))
            model_times.append(('Warm-MIOSR-Opt', sum(warm_mio_model.optimizer.solve_times)))
            model_times.append(('Warm-MIOSR-Build', sum(warm_mio_model.optimizer.build_times)))

        if seed == 1:  # For logging
            print(system.name, duration, model_times)

        for model_name, runtime in model_times:
            performance_metrics = {
                'runtime': runtime
            }

            key = (model_name, duration, dt, noise_fraction, seed, library_dim, tuple(x0))
            results[key] = performance_metrics

    result_df = pd.DataFrame(results).T
    cols = list(result_df.columns)
    result_df = result_df.reset_index()
    result_df.columns = ['Method', 'Duration', 'Sample Rate', 'Noise', 'Seed', 'Dim', 'x0'] + cols
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', required=True)
    parser.add_argument('-t', '--trial', required=True, type=int)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-p', '--polyorder', required=True, type=int)
    parser.add_argument('-w', '--run_warmstart', dest='run_warmstart', default=False, action='store_true')

    args = vars(parser.parse_args())

    start_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
    print(f"Starting trial {args['trial']} at {start_time}")

    if args['system'].lower() == 'lorenz':
        results = runtime_experiment(
            system=Lorenz(),
            seed=args['trial'],
            durations=2**np.linspace(-1, 7, 12),
            noise_fraction=0.002,
            dt=0.002,
            group_sparsity=(2, 3, 2),
            poly_order=args['polyorder'],
            smooth_window_length=9,
            run_warmstart=args['run_warmstart']
        )
    elif args['system'].lower() == 'hopf':
        results = runtime_experiment(
            system=Hopf(),
            seed=args['trial'],
            durations=2 ** np.linspace(-1, 7, 12),
            noise_fraction=0.002,
            dt=0.002,
            threshold=0.01,
            group_sparsity=(4, 4),
            poly_order=args['polyorder'],
            smooth_window_length=9,
            run_warmstart=args['run_warmstart'],
        )
    elif args['system'].lower()[:3] == 'mhd':
        durations = 2 ** np.linspace(-1, 7, 12)
        if args['system'].lower() == 'mhde' and args['polyorder'] == 5:
            durations = durations[::2]
        elif args['system'].lower() == 'mhdo' and args['polyorder'] == 5:
            durations = durations[1::2]
        results = runtime_experiment(
            system=MHD(),
            seed=args['trial'],
            durations=durations,
            noise_fraction=0.002,
            dt=0.002,
            threshold=0.2,
            group_sparsity=(2, 2, 2, 2, 2, 2),
            poly_order=args['polyorder'],
            smooth_window_length=9,
            run_warmstart=args['run_warmstart'],
        )
    elif args['system'].lower() == 'test':
        results = runtime_experiment(
            system=Lorenz(),
            seed=args['trial'],
            durations=2 ** np.linspace(3, 4, 2),
            noise_fraction=0.002,
            dt=0.002,
            group_sparsity=(2, 3, 2),
            poly_order=args['polyorder'],
            smooth_window_length=9,
            run_warmstart=args['run_warmstart'],
        )
    else:
        raise ValueError('Not a valid system')

    result_root_dir = os.path.join(os.getcwd().split('sindy_mio')[0], 'sindy_mio', 'results')
    experiment_name = f'{args["name"]}_{args["system"]}_{args["polyorder"]}'
    experiment_path = os.path.join(result_root_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    result_file = os.path.join(experiment_path, f'trial_{args["trial"]}.csv')
    results.to_csv(result_file, index=False)
