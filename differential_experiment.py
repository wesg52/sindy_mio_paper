import argparse
import datetime
import pandas as pd
from pysindy.differentiation import SmoothedFiniteDifference
import os
from sklearn.metrics import mean_squared_error
import time

from evaluation_metrics import *
from miosr import *
from models import *
from systems import *


def differential_multi_algorithm_benchmark(
        system,
        seed=0,
        durations=(0.5, 1, 2, 4, 8),
        noise_levels=(0.001, 0.01),
        sample_rates=(0.001,),
        thresholds=10**np.linspace(-3, 0, 20),
        sr3_thresholds=10**np.linspace(-2, 1, 20),
        alphas=(0, 1e-5, 1e-3, 0.01, 0.05, 0.2),
        k_span=(1, 5),
        validation_fraction=0.5,
        poly_order=5,
        smooth_window_length=9,
        n_ensemble_models=50,
):
    feature_library = ps.PolynomialLibrary(degree=poly_order)
    differentiation_method = SmoothedFiniteDifference(
        smoother_kws={'window_length': smooth_window_length}
    )
    results = {}
    for duration, noise_fraction, dt in product(durations, noise_levels, sample_rates):
        x0 = system.sample_initial_conditions(n=1, seed=seed)[0]
        t, x = system.simulate(duration * (1 + validation_fraction), dt, x0=x0)
        library_dim = feature_library.fit_transform(x).shape[1]
        rmse = mean_squared_error(x, np.zeros(x.shape), squared=False)
        np.random.seed(seed)
        x_noised = x + np.random.normal(0, rmse * noise_fraction, x.shape)
        valid_ix = int(duration / dt)
        x_train, x_valid = x_noised[:valid_ix, :], x_noised[valid_ix:, :]

        start_t = time.time()
        stlsq_model, stlsq_params = fit_and_tune_stlsq(
            feature_library,
            differentiation_method,
            x_train,
            dt,
            x_valid,
            thresholds,
            alphas)
        stlsq_time = time.time()

        best_threshold, best_alpha = [param[1] for param in stlsq_params]
        ensemble_model, ensemble_params = fit_and_tune_ensemble_stlsq(
            feature_library,
            differentiation_method,
            x_train,
            dt,
            x_valid,
            n_ensemble_models,
            thresholds,
            [best_alpha])
        ensemble_time = time.time()

        sr3_model, sr3_params = fit_and_tune_sr3(
            feature_library,
            differentiation_method,
            x_train,
            dt,
            x_valid,
            sr3_thresholds)
        sr3_time = time.time()

        ssr_model, ssr_params = fit_and_tune_ssr(
            feature_library,
            differentiation_method,
            library_dim,
            x_train,
            dt,
            x_valid,
            alphas
        )
        ssr_time = time.time()

        mio_sos_model, mio_sos_params = fit_and_tune_miosos(
            feature_library,
            differentiation_method,
            x_train,
            dt,
            x_valid,
            k_span,
            alphas
        )
        mio_time = time.time()

        models = [
            ('STLSQ', stlsq_model, stlsq_params, stlsq_time - start_t),
            ('E-STLSQ', ensemble_model, ensemble_params, ensemble_time - stlsq_time),
            ('SR3', sr3_model, sr3_params, sr3_time - ensemble_time),
            ('SSR', ssr_model, ssr_params, ssr_time - sr3_time),
            ('MIOSR', mio_sos_model, mio_sos_params, mio_time - ssr_time)
        ]

        for model_name, model, model_params, runtime in models:
            learned_coefs = model.optimizer.coef_
            true_coefs = system.true_coefs
            tp, fp, fn = support_confusion_matrix(true_coefs, learned_coefs)
            test_r2s, test_mses = clean_test_scores(model, system, seed=seed+1)  # Don't duplicate initial cond
            performance_metrics = {
                'insample_r2': model.score(x_train, t=dt),
                'validation_r2': model.score(x_valid, t=dt),
                'validation_mse': model.score(x_valid, t=dt, metric=mean_squared_error),
                'test_r2': np.mean(test_r2s),
                'test_rmse': np.mean(test_mses),
                'coef_l1': coefficient_normalized_l1(true_coefs, learned_coefs),
                'coef_l2': coefficient_normalized_l2(true_coefs, learned_coefs),
                'coef_linf': coefficient_normalized_linf(true_coefs, learned_coefs),
                'support_accuracy': support_true_positivity(true_coefs, learned_coefs),
                'support_tp': tp,
                'support_fp': fp,
                'support_fn': fn,
                'runtime': runtime
            }
            algo_metrics = {}
            for k, v in model_params:
                algo_metrics[k] = v
            if model_name[:6] == "MIOSOS":
                algo_metrics['objVal'] = model.optimizer._model.objVal
                try:
                    algo_metrics['MIPGap'] = model.optimizer._model.MIPGap
                except AttributeError:
                    algo_metrics['MIPGap'] = 'unavailable'

            key = (model_name, duration, dt, noise_fraction, seed, tuple(x0))
            results[key] = {**performance_metrics, **algo_metrics}

    result_df = pd.DataFrame(results).T
    cols = list(result_df.columns)
    result_df = result_df.reset_index()
    result_df.columns = ['Method', 'Duration', 'Sample Rate', 'Noise', 'Seed', 'x0'] + cols
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', required=True)
    parser.add_argument('-t', '--trial', required=True, type=int)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-o', '--polyorder', type=int, default=5)
    args = vars(parser.parse_args())

    start_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
    print(f'Starting trial {args["trial"]} at {start_time}')
    poly_order = args['polyorder']
    if args['system'].lower() == 'lorenz':
        results = differential_multi_algorithm_benchmark(
            system=Lorenz(library_size=get_polylib_size(poly_order, 3)),
            seed=args['trial'],
            thresholds=10**np.linspace(-2, 1, 50),
            sr3_thresholds=10**np.linspace(-1.5, 1.5, 50),
            durations=2**np.linspace(-2, 3.4, 12),
            noise_levels=(0.002,),
            sample_rates=(0.002,),
            k_span=(1, 5),
            poly_order=poly_order,
            smooth_window_length=9,
        )
    elif args['system'].lower() == 'hopf':
        results = differential_multi_algorithm_benchmark(
            system=Hopf(library_size=get_polylib_size(poly_order, 2)),
            seed=args['trial'],
            thresholds=10**np.linspace(-3, 0, 50),
            sr3_thresholds=10 ** np.linspace(-3.5, 0.5, 50),
            durations=2**np.linspace(0, 4, 12),
            noise_levels=(0.002,),
            sample_rates=(0.002,),
            k_span=(1, 5),
            poly_order=poly_order,
            smooth_window_length=9,
        )
    elif args['system'].lower() == 'mhd':
        results = differential_multi_algorithm_benchmark(
            system=MHD(library_size=get_polylib_size(poly_order, 6)),
            seed=args['trial'],
            durations=2**np.linspace(-1, 4, 12),
            thresholds=10**np.linspace(-1.5, 1.5, 50),
            sr3_thresholds=10 ** np.linspace(-2, 2, 50),
            noise_levels=(0.002,),
            sample_rates=(0.002,),
            k_span=(1, 5),
            poly_order=poly_order,
            smooth_window_length=9,
        )
    elif args['system'].lower() == 'test':
        results = differential_multi_algorithm_benchmark(
            system=Lorenz(library_size=get_polylib_size(poly_order, 3)),
            seed=args['trial'],
            durations=(4, 8),
            noise_levels=(0.001,),
            sample_rates=(0.001,),
            alphas=(0.01, 0.05),
            k_span=(2, 3),
            poly_order=args['polyorder'],
            smooth_window_length=9,
        )
    else:
        raise ValueError('Not a valid system')

    result_root_dir = os.path.join(os.getcwd().split('sindy_mio')[0], 'sindy_mio', 'results')
    experiment_name = f'{args["name"]}_{args["system"]}'
    experiment_path = os.path.join(result_root_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    result_file = os.path.join(experiment_path, f'trial_{args["trial"]}.csv')
    results.to_csv(result_file, index=False)
