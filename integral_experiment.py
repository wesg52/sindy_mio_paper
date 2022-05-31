import argparse
import datetime
import os

import pandas as pd
from pysindy.feature_library import *

from systems import *
from weak_models import *


def reindex_polylibrary(x_train, lib_names, degree=3):
    lib = ps.PolynomialLibrary(degree=degree)
    lib.fit_transform(x_train)
    pln = lib.get_feature_names()
    pln_map = {k: i for i, k in enumerate(pln)}
    index_ordering = [pln_map[name] for name in lib_names]
    return index_ordering


def integral_multi_algorithm_benchmark(
        system,
        seed=0,
        durations=(30,),
        noise_levels=(0.001, 0.01, 0.10),
        dt=0.002,
        thresholds=10**np.linspace(-1, 2, 20),
        alphas=(0, 1e-5, 0.001, 0.01, 0.05),
        k_span=(1, 5),
        validation_fraction=0.5,
        library_functions=None,
        library_function_names=None,
        library_pts_per_domain=(100,),
        library_n_domains=(100,),
        n_ensemble_models=50,
):

    results = {}
    for duration, noise_fraction, pts_per_domain, n_domains \
            in product(durations, noise_levels, library_pts_per_domain, library_n_domains):
        x0 = system.sample_initial_conditions(n=1, seed=seed)[0]
        t, x = system.simulate(duration * (1 + validation_fraction), dt, x0=x0)
        rmse = mean_squared_error(x, np.zeros(x.shape), squared=False)
        np.random.seed(seed)
        x_noised = x + np.random.normal(0, rmse * noise_fraction, x.shape)
        valid_ix = int(duration / dt)
        x_train, x_valid = x_noised[:valid_ix, :], x_noised[valid_ix:, :]

        ode_lib = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            spatiotemporal_grid=t[:valid_ix],
            is_uniform=True,
            include_bias=True,
            num_pts_per_domain=pts_per_domain,
            K=n_domains,
        )
        valid_lib = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            spatiotemporal_grid=t[:len(x_valid)],
            is_uniform=True,
            include_bias=True,
            num_pts_per_domain=pts_per_domain,
            K=n_domains,
        )
        valid_lib.fit(x_valid)
        u_dot_integral = convert_u_dot_integral(x_train, ode_lib)
        dummy_model = ps.SINDy(feature_library=ode_lib)
        dummy_model.fit(x_train)
        theta = copy.deepcopy(dummy_model.optimizer.Theta_)
        id_lib = IdentityLibrary()
        cached_features = (u_dot_integral, theta, id_lib)

        use_weak_xdot_for_validation = noise_fraction > 0.15

        start_t = time.time()
        stlsq_model, stlsq_params = fit_and_tune_weak_stlsq(
            cached_features,
            ode_lib,
            valid_lib,
            x_train,
            dt,
            x_valid,
            thresholds,
            alphas,
            use_weak_xdot_for_validation=use_weak_xdot_for_validation)
        stlsq_time = time.time() - start_t
        ensemble_alpha_search = [stlsq_params[1][1]]

        start_t = time.time()
        ensemble_model, ensemble_params = fit_and_tune_weak_ensemble_stlsq(
            cached_features,
            ode_lib,
            valid_lib,
            x_train,
            dt,
            x_valid,
            thresholds,
            ensemble_alpha_search,
            n_ensemble_models=n_ensemble_models,
            use_weak_xdot_for_validation=use_weak_xdot_for_validation)
        ensemble_time = time.time() - start_t

        start_t = time.time()
        ssr_model, ssr_params = fit_and_tune_weak_ssr(
            cached_features,
            ode_lib,
            valid_lib,
            x_train,
            dt,
            x_valid,
            alphas,
            use_weak_xdot_for_validation=use_weak_xdot_for_validation)
        ssr_time = time.time() - start_t

        start_t = time.time()
        sr3_model, sr3_params = fit_and_tune_weak_sr3(
            cached_features,
            ode_lib,
            valid_lib,
            x_train,
            dt,
            x_valid,
            thresholds,
            use_weak_xdot_for_validation=use_weak_xdot_for_validation)
        sr3_time = time.time() - start_t

        start_t = time.time()
        mio_sos_model, mio_sos_params = fit_and_tune_weak_miosr(
            cached_features,
            ode_lib,
            valid_lib,
            x_train,
            dt,
            x_valid,
            k_span,
            alphas,
            use_weak_xdot_for_validation=use_weak_xdot_for_validation
        )
        mio_time = time.time() - start_t

        if seed == 1:
            print(system.name, duration, noise_fraction, pts_per_domain, n_domains, mio_time)

        models = [
            ('STLSQ', stlsq_model, stlsq_params, stlsq_time),
            ('SSR', ssr_model, ssr_params, ssr_time),
            ('SR3', sr3_model, sr3_params, sr3_time),
            ('E-STLSQ', ensemble_model, ensemble_params, ensemble_time),
            ('MIOSR', mio_sos_model, mio_sos_params, mio_time),
        ]

        lib_index = reindex_polylibrary(x_train, ode_lib.get_feature_names())
        for model_name, model, model_params, runtime in models:
            learned_coefs = model.optimizer.coef_
            true_coefs = system.true_coefs[:, lib_index]
            tp, fp, fn = support_confusion_matrix(true_coefs, learned_coefs)
            test_r2s, test_mses = clean_test_scores(model, system, seed=seed+1)  # Don't duplicate initial cond
            performance_metrics = {
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
                algo_metrics['MIPGap'] = model.optimizer._model.MIPGap

            key = (model_name, duration, dt, noise_fraction, pts_per_domain, n_domains, seed, tuple(x0))
            results[key] = {**performance_metrics, **algo_metrics}

    result_df = pd.DataFrame(results).T
    cols = list(result_df.columns)
    result_df = result_df.reset_index()
    result_df.columns = ['Method', 'Duration', 'Sample Rate', 'Noise', 'Pts/Domain', '# Domains', 'Seed', 'x0'] + cols
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', required=True)
    parser.add_argument('-t', '--trial', required=True, type=int)
    parser.add_argument('-n', '--name', required=True)
    args = vars(parser.parse_args())

    start_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
    print(f'Starting trial {args["trial"]} at {start_time}')

    library_functions = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x, y: x * y,
        lambda x, y: x ** 2 * y,
        lambda x, y: x * y ** 2,
        lambda x, y, z: x * y * z

    ]
    library_function_names = [
        lambda x: x,
        lambda x: f'{x}^2',
        lambda x: f'{x}^3',
        lambda x, y: f'{x} {y}',
        lambda x, y: f'{x}^2 {y}',
        lambda x, y: f'{x} {y}^2',
        lambda x, y, z: f'{x} {y} {z}',

    ]

    if args['system'].lower() == 'lotka':
        results = integral_multi_algorithm_benchmark(
            system=Lotka(library_size=10),
            seed=args['trial'],
            durations=(50, ),
            noise_levels=10 ** np.linspace(-2, -0.3, 12),
            dt=0.002,
            thresholds=2**np.linspace(-3, 4, 50),
            k_span=(1, 5),
            library_functions=library_functions[:-1],
            library_function_names=library_function_names[:-1],
            library_pts_per_domain=(400,),
            library_n_domains=(2400,)
        )
    elif args['system'].lower() == 'vanderpol':
        results = integral_multi_algorithm_benchmark(
            system=VanderPol(library_size=10, p=3),
            seed=args['trial'],
            durations=(50, ),
            noise_levels=10 ** np.linspace(-2, -0.3, 12),
            dt=0.002,
            thresholds=2**np.linspace(-1, 5, 50),
            k_span=(1, 5),
            library_functions=library_functions[:-1],
            library_function_names=library_function_names[:-1],
            library_pts_per_domain=(400,),
            library_n_domains=(2400,)
        )
    elif args['system'].lower() == 'rossler':
        results = integral_multi_algorithm_benchmark(
            system=Rossler(library_size=20),
            seed=args['trial'],
            durations=(50,),
            noise_levels=10 ** np.linspace(-2, -0.3, 12),
            dt=0.002,
            thresholds=2**np.linspace(2, 6, 50),
            k_span=(1, 5),
            library_functions=library_functions,
            library_function_names=library_function_names,
            library_pts_per_domain=(400, ),
            library_n_domains=(2400,)
        )
    elif args['system'].lower() == 'test':
        results = integral_multi_algorithm_benchmark(
            system=Lotka(library_size=10),
            seed=args['trial'],
            durations=(50,),
            thresholds=np.linspace(3, 10, 10),
            alphas=(0.05, ),
            noise_levels=(0.01, 0.2),
            dt=0.002,
            k_span=(1, 5),
            library_functions=library_functions[:-1],
            library_function_names=library_function_names[:-1],
            library_pts_per_domain=(100,),
            library_n_domains=(1600,)
        )
    else:
        raise ValueError('Not a valid system')

    result_root_dir = os.path.join(os.getcwd().split('sindy_mio')[0], 'sindy_mio', 'results')
    experiment_name = f'{args["name"]}_{args["system"]}'
    experiment_path = os.path.join(result_root_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    result_file = os.path.join(experiment_path, f'trial_{args["trial"]}.csv')
    results.to_csv(result_file, index=False)
