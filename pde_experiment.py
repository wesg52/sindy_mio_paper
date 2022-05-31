import argparse
import datetime
import pandas as pd
from pysindy.differentiation import SmoothedFiniteDifference
import os
from sklearn.metrics import mean_squared_error
import time

from evaluation_metrics import *
from miosr import *
from weak_models import *
from systems import *
from pysindy.feature_library import *
from pysindy.utils import convert_u_dot_integral


def reindex_polylibrary(x_train, lib_names, degree=3):
    lib = ps.PolynomialLibrary(degree=degree)
    lib.fit_transform(x_train)
    pln = lib.get_feature_names()
    pln_map = {k: i for i, k in enumerate(pln)}
    index_ordering = [pln_map[name] for name in lib_names]
    return index_ordering


def pde_experiment(
        system,
        seed=0,
        durations=(30,),
        noise_levels=(0.001, 0.01, 0.1),
        dt=0.002,
        thresholds=(0.1, ),
        stlsq_alphas=(0, 1e-5, 1e-3),
        mio_alphas=(1e-5,),
        sr3_nus=(1 / 30, 0.1, 1 / 3, 1, 10 / 3),
        n_candidates_to_drop=2,
        n_ensemble_models=50,
        derivative_order=2,
        regression_timeout=30,
        is_periodic=True,
        group_sparsity=(1, 5),
        library_functions=None,
        library_function_names=None,
        library_pts_per_domain=(100,),
        library_n_domains=(100,)
):
    results = {}
    for duration, noise_fraction, pts_per_domain, n_domains \
            in product(durations, noise_levels, library_pts_per_domain, library_n_domains):

        x0 = system.sample_initial_conditions(n=1, seed=seed)[0]
        u = system.simulate(duration, dt, x0=x0)
        rmse = (np.mean(u ** 2, axis=0) ** 0.5).mean()
        np.random.seed(seed)
        u_train = u + np.random.normal(0, rmse * noise_fraction, u.shape)

        spatiotemporal_grid = system.make_mesh_grid(duration, dt)
        weak_lib = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            derivative_order=derivative_order,
            spatiotemporal_grid=spatiotemporal_grid,
            K=n_domains,
            is_uniform=True,
            num_pts_per_domain=pts_per_domain,
            periodic=is_periodic
        )
        u_dot_integral = convert_u_dot_integral(u_train, weak_lib)
        dummy_model = ps.SINDy(feature_library=weak_lib)
        dummy_model.fit(u_train)
        theta = copy.deepcopy(dummy_model.optimizer.Theta_)
        id_lib = IdentityLibrary()

        start_t = time.time()
        coefs = []
        for alpha in mio_alphas:
            mio_model = ps.SINDy(
                optimizer=MIOSR(
                    group_sparsity=group_sparsity,
                    alpha=alpha,
                    normalize_columns=True,
                    regression_timeout=regression_timeout,
                ),
                feature_library=id_lib,
            )
            mio_model.fit(theta, x_dot=u_dot_integral, quiet=True)
            coefs.append(mio_model.coefficients())
        coef_errors = [support_true_positivity(system.true_coefs, coef) for coef in coefs]
        best_ix = np.argmax(np.array(coef_errors))
        mio_model.optimizer.coef_ = coefs[best_ix]
        mio_time = time.time() - start_t

        start_t = time.time()
        coefs = []
        params = list(product(stlsq_alphas, thresholds))
        for alpha, threshold in params:
            stlsq_model = ps.SINDy(
                optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True),
                feature_library=id_lib,
            )
            stlsq_model.fit(theta, x_dot=u_dot_integral, quiet=True)
            coefs.append(stlsq_model.coefficients())
        coef_errors = [support_true_positivity(system.true_coefs, coef) for coef in coefs]
        best_ix = np.argmax(np.array(coef_errors))
        opt_alpha, opt_threshold = params[best_ix]
        stlsq_model.optimizer.coef_ = coefs[best_ix]
        stlsq_time = time.time() - start_t

        start_t = time.time()
        coefs = []
        for alpha in stlsq_alphas:
            ssr_model = ps.SINDy(
                optimizer=ps.SSR(max_iter=theta.shape[1] - 1, alpha=alpha, normalize_columns=True),
                feature_library=id_lib,
            )
            ssr_model.fit(theta, x_dot=u_dot_integral, quiet=True)
            for j in range(len(ssr_model.optimizer.history_)):
                coefs.append(ssr_model.optimizer.history_[j])
        coef_errors = [support_true_positivity(system.true_coefs, coef) for coef in coefs]
        best_ix = np.argmax(np.array(coef_errors))
        ssr_model.optimizer.coef_ = coefs[best_ix]
        ssr_time = time.time() - start_t

        start_t = time.time()
        coefs = []
        params = list(product(sr3_nus, thresholds))
        for nu, threshold in params:
            sr3_model = ps.SINDy(
                optimizer=ps.SR3(threshold=threshold, nu=nu, max_iter=10000, normalize_columns=True),
                feature_library=id_lib,
            )
            sr3_model.fit(theta, x_dot=u_dot_integral, quiet=True)
            coefs.append(sr3_model.coefficients())
        coef_errors = [support_true_positivity(system.true_coefs, coef) for coef in coefs]
        best_ix = np.argmax(np.array(coef_errors))
        sr3_model.optimizer.coef_ = coefs[best_ix]
        sr3_time = time.time() - start_t

        start_t = time.time()
        ensemble_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=opt_threshold, alpha=opt_alpha, normalize_columns=True),
            feature_library=id_lib,
        )
        ensemble_model.fit(theta, x_dot=u_dot_integral, quiet=True,
                           library_ensemble=True,
                           n_models=n_ensemble_models,
                           n_candidates_to_drop=n_candidates_to_drop)
        ensemble_time = time.time() - start_t
        coef_order = (np.abs(np.array(ensemble_model.coef_list)) > 0).mean(axis=0).argsort(axis=1)
        coef_medians = np.median(ensemble_model.coef_list, axis=0)
        for ix, k in enumerate(group_sparsity):
            coef_medians[ix, coef_order[ix, :-k]] = 0
        ensemble_model.optimizer.coef_ = coef_medians


        if seed == 1:
            print(system.name, duration, noise_fraction, pts_per_domain, n_domains, mio_time)

        models = [
            ('MIOSR', mio_model, mio_time),
            ('STLSQ', stlsq_model, stlsq_time),
            ('SSR', ssr_model, ssr_time),
            ('SR3', sr3_model, sr3_time),
            ('E-STLSQ', ensemble_model, ensemble_time),
        ]

        for model_name, model, runtime in models:
            learned_coefs = model.optimizer.coef_
            true_coefs = system.true_coefs
            tp, fp, fn = support_confusion_matrix(true_coefs, learned_coefs)
            performance_metrics = {
                'coef_l1': coefficient_normalized_l1(true_coefs, learned_coefs),
                'coef_l2': coefficient_normalized_l2(true_coefs, learned_coefs),
                'coef_linf': coefficient_normalized_linf(true_coefs, learned_coefs),
                'support_accuracy': support_true_positivity(true_coefs, learned_coefs),
                'support_tp': tp,
                'support_fp': fp,
                'support_fn': fn,
                'runtime': runtime
            }
            key = (model_name, duration, dt, noise_fraction, pts_per_domain, n_domains, seed)
            results[key] = performance_metrics

    result_df = pd.DataFrame(results).T
    cols = list(result_df.columns)
    result_df = result_df.reset_index()
    result_df.columns = ['Method', 'Duration', 'Sample Rate', 'Noise', 'Pts/Domain', '# Domains', 'Seed'] + cols
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

    if args['system'].lower() == 'ks':
        results = pde_experiment(
            system=KuramotoSivashinsky(),
            seed=args['trial'],
            durations=(25, ),
            noise_levels=np.linspace(0.25, 3, 12),
            dt=0.1,
            thresholds=np.linspace(0.4, 2.0, 20),
            stlsq_alphas=(0, 1e-5, 0.001, 0.01, 0.05),
            mio_alphas=(0, 1e-5, 0.001, 0.01, 0.05),
            n_candidates_to_drop=2,
            n_ensemble_models=50,
            derivative_order=4,
            group_sparsity=(3,),
            library_functions=library_functions[:3],
            library_function_names=library_function_names[:3],
            library_pts_per_domain=(50, ),
            library_n_domains=(200, )
        )
    elif args['system'].lower() == 'ksgrid':
        results = pde_experiment(
            system=KuramotoSivashinsky(),
            seed=args['trial'],
            durations=np.linspace(5, 50, 10),
            noise_levels=np.linspace(0.25, 3, 12),
            dt=0.1,
            thresholds=2**np.linspace(-4, 3, 30),
            stlsq_alphas=(0.001, 0.01, 0.05),
            n_candidates_to_drop=2,
            n_ensemble_models=50,
            derivative_order=4,
            group_sparsity=(3,),
            library_functions=library_functions[:3],
            library_function_names=library_function_names[:3],
            library_pts_per_domain=(50, ),
            library_n_domains=(200, )
        )
    elif args['system'].lower() == 'redif':
        results = pde_experiment(
            system=ReactionDiffusion(grid_size=256),
            seed=args['trial'],
            durations=(5,),
            noise_levels=np.linspace(.025, 0.3, 12),
            dt=0.02,
            thresholds=np.linspace(0.04, 0.16, 20),
            stlsq_alphas=(0, 1e-5, 0.001, 0.01, 0.05),
            n_candidates_to_drop=3,
            n_ensemble_models=50,
            derivative_order=2,
            regression_timeout=300,
            group_sparsity=(7, 7),
            library_functions=library_functions[:-1],
            library_function_names=library_function_names[:-1],
            library_pts_per_domain=(36,),
            library_n_domains=(400,)
        )
    elif args['system'].lower() == 'test':
        results = pde_experiment(
            system=KuramotoSivashinsky(),
            seed=args['trial'],
            durations=(50,),
            noise_levels=(0.1, 0.5),
            derivative_order=4,
            thresholds=(1, ),
            stlsq_alphas=(0.005, ),
            n_candidates_to_drop=1,
            n_ensemble_models=10,
            dt=0.1,
            group_sparsity=(3,),
            library_functions=library_functions[:3],
            library_function_names=library_function_names[:3],
            library_pts_per_domain=(50,),
            library_n_domains=(200,)
        )
    else:
        raise ValueError('Not a valid system')

    result_root_dir = os.path.join(os.getcwd().split('sindy_mio')[0], 'sindy_mio', 'results')
    experiment_name = f'{args["name"]}_{args["system"]}'
    experiment_path = os.path.join(result_root_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    result_file = os.path.join(experiment_path, f'trial_{args["trial"]}.csv')
    results.to_csv(result_file, index=False)
