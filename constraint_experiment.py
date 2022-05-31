import argparse
import datetime
import pandas as pd
from pysindy.differentiation import SmoothedFiniteDifference
import os
from sklearn.metrics import mean_squared_error
import time

from evaluation_metrics import *
from miosr import *
from systems import *


def clean_duffing_test_scores(model, system, n=10):
    r2s = []
    mses = []
    x0s = system.sample_initial_conditions(n)
    for x0 in x0s:
        t, x_test = system.simulate(x0=x0)
        diff = ps.FiniteDifference()
        x_dot_test = diff._differentiate(x_test, t)
        r2s.append(model.score(x_test[:, :2], x_dot=x_dot_test[:, 2:]))
        mses.append(model.score(x_test[:, :2], x_dot=x_dot_test[:, 2:], metric=mean_squared_error))
    return np.array(r2s), np.sqrt(np.array(mses))


def fit_and_tune_miosos_constrained(feature_library, dif_method, x_train, x_valid, x_dot_train, x_dot_valid,
                        sparsity_search=(1, 10), alphas=(0,), constraint_lhs=None, constraint_rhs=None):
    aics = []
    coef_history = []
    params = []
    for k, alpha in product(range(sparsity_search[0], sparsity_search[1] + 1), alphas):
        mio_model = ps.SINDy(
            optimizer=MIOSR(target_sparsity=k, alpha=alpha,
                            constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs,
                            constraint_order='feature'),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        mio_model.fit(x_train, x_dot=x_dot_train, quiet=True, unbias=False)
        coef_history.append(mio_model.optimizer.coef_)
        params.append((k, alpha))

        x_dot_pred = mio_model.predict(x_valid)
        k = (np.abs(mio_model.coefficients()) > 0).sum()

        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)

    min_aic = np.argmin(np.array(aics))
    coef_history = np.array(coef_history)
    best_model = coef_history[min_aic]
    mio_model.optimizer.coef_ = best_model
    mio_model.optimizer.target_sparsity = params[min_aic][0]
    return mio_model, (('group_sparsity', params[min_aic][0]), ('alpha', params[min_aic][1]))


def fit_and_tune_sr3(feature_library, dif_method, x_train, x_valid, x_dot_train, x_dot_valid,
                     thresholds,
                     nus=(1/30, 0.1, 1/3, 1, 10/3),
                     constraint_rhs=None,
                     constraint_lhs=None):
    aics = []
    coefs = []
    params = list(product(thresholds, nus))
    for threshold, nu in params:
        if constraint_lhs is None and constraint_rhs is None:
            model = ps.SINDy(
                optimizer=ps.SR3(threshold=threshold, nu=nu, max_iter=10000),
                feature_library=feature_library,
                differentiation_method=dif_method
            )
        else:
            model = ps.SINDy(
                optimizer=ps.ConstrainedSR3(threshold=threshold, nu=nu, max_iter=10000,
                                            constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs,
                                            constraint_order='feature'),
                feature_library=feature_library,
                differentiation_method=dif_method
            )
        model.fit(x_train, x_dot=x_dot_train, quiet=True, unbias=False)

        x_dot_pred = model.predict(x_valid)
        k = (np.abs(model.coefficients()) > 0).sum()
        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)
        coefs.append(model.coefficients())

    best_model_ix = np.argmin(np.array(aics))
    best_t, best_nu = params[best_model_ix]
    model.optimizer.coef_ = coefs[best_model_ix]
    return model, (('threshold', best_t), ('nu', best_nu))


def constraint_benchmark(
        system,
        seed=0,
        durations=(0.5, 1, 2, 4, 8),
        noise_levels=(0.001, 0.01),
        sample_rates=(0.001,),
        thresholds=10**np.linspace(-3, 0, 20),
        sparsity_search=(1, 5),
        validation_fraction=0.5,
        poly_order=3,
        smooth_window_length=21,
):
    feature_library = ps.PolynomialLibrary(degree=poly_order, include_bias=True)
    differentiation_method = SmoothedFiniteDifference(
        smoother_kws={'window_length': smooth_window_length}
    )

    n_constraints = 6
    l = 10
    gradient_constraint = np.zeros((n_constraints, 2 * l))
    gradient_constraint[0, 3] = 1
    gradient_constraint[0, 4] = -1
    gradient_constraint[1, 7] = 2
    gradient_constraint[1, 8] = -1
    gradient_constraint[2, 9] = 1
    gradient_constraint[2, 10] = -2
    gradient_constraint[3, 13] = 3
    gradient_constraint[3, 14] = -1
    gradient_constraint[4, 15] = 1
    gradient_constraint[4, 16] = -1
    gradient_constraint[5, 17] = 1
    gradient_constraint[5, 18] = -3
    constraint_rhs = np.zeros((n_constraints))

    results = {}
    for duration, noise_fraction, dt in product(durations, noise_levels, sample_rates):
        x0 = system.sample_initial_conditions(n=1, seed=seed)[0]
        t, x = system.simulate(duration * (1 + validation_fraction), dt, x0=x0)
        rmse = mean_squared_error(x[:, :2], np.zeros(x[:, :2].shape), squared=False)
        np.random.seed(seed)
        x_noised = x + np.random.normal(0, rmse * noise_fraction, x.shape)
        x_dot = differentiation_method._differentiate(x_noised, t)
        valid_ix = int(duration / dt)
        # Assumes 2D duffing setup
        x_train, x_valid = x_noised[:valid_ix, :2], x_noised[valid_ix:, :2]
        x_dot_train, x_dot_valid = x_dot[:valid_ix, 2:], x_dot[valid_ix:, 2:]


        start_t = time.time()
        sr3_model, sr3_params = fit_and_tune_sr3(
            feature_library, differentiation_method, x_train, x_valid, x_dot_train, x_dot_valid,
            thresholds,
            nus=(1 / 30, 0.1, 1 / 3, 1, 10 / 3),
            constraint_rhs=None,
            constraint_lhs=None)
        sr3_time = time.time() - start_t

        start_t = time.time()
        con_sr3_model, con_sr3_params = fit_and_tune_sr3(
            feature_library, differentiation_method, x_train, x_valid, x_dot_train, x_dot_valid,
            thresholds,
            nus=(1 / 30, 0.1, 1 / 3, 1, 10 / 3),
            constraint_rhs=constraint_rhs,
            constraint_lhs=gradient_constraint
        )
        con_sr3_time = time.time() - start_t

        start_t = time.time()
        mio_sos_model, mio_params = fit_and_tune_miosos_constrained(
            feature_library, differentiation_method, x_train, x_valid, x_dot_train, x_dot_valid,
            sparsity_search=sparsity_search, alphas=(0.0001, 0.001, 0.01), constraint_lhs=None, constraint_rhs=None
        )
        mio_time = time.time() - start_t

        start_t = time.time()
        con_mio_sos_model, con_mio_params = fit_and_tune_miosos_constrained(
            feature_library, differentiation_method, x_train, x_valid, x_dot_train, x_dot_valid,
            sparsity_search=sparsity_search, alphas=(0.0001, 0.001, 0.01),
            constraint_rhs=constraint_rhs,
            constraint_lhs=gradient_constraint
        )
        con_mio_time = time.time() - start_t

        models = [
            ('SR3', sr3_model, sr3_params, sr3_time),
            ('MIOSR', mio_sos_model, mio_params, mio_time),
            ('Con-SR3', con_sr3_model, con_sr3_params, con_sr3_time),
            ('Con-MIOSR', con_mio_sos_model, con_mio_params, con_mio_time)
        ]

        for model_name, model, model_params, runtime in models:
            learned_coefs = model.optimizer.coef_
            true_coefs = system.true_coefs
            tp, fp, fn = support_confusion_matrix(true_coefs, learned_coefs)
            test_r2s, test_mses = clean_duffing_test_scores(model, system)
            performance_metrics = {
                'insample_r2': model.score(x_train, x_dot=x_dot_train),
                'validation_r2': model.score(x_valid, x_dot=x_dot_valid),
                'validation_mse': model.score(x_valid, x_dot=x_dot_valid, metric=mean_squared_error),
                'test_r2': np.mean(test_r2s),
                'test_rmse': np.mean(test_mses),
                'coef_l1': coefficient_normalized_l1(true_coefs, learned_coefs),
                'coef_l2': coefficient_normalized_l2(true_coefs, learned_coefs),
                'coef_linf': coefficient_normalized_linf(true_coefs, learned_coefs),
                'support_accuracy': support_true_positivity(true_coefs, learned_coefs),
                'support_tp': tp,
                'support_fp': fp,
                'support_fn': fn,
                'constraint_violation': np.mean(np.abs(constraint_violation(model, gradient_constraint))),
                'runtime': runtime
            }
            algo_metrics = {}
            for k, v in model_params:
                algo_metrics[k] = v
            if model_name[:6] == "MIOSOS":
                algo_metrics['objVal'] = model.optimizer._model.objVal
                algo_metrics['MIPGap'] = model.optimizer._model.MIPGap

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
    args = vars(parser.parse_args())

    start_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
    print(f'Starting trial {args["trial"]} at {start_time}')
    if args['system'].lower() == 'duffing':
        results = constraint_benchmark(
            system=Duffing(library_size=10, p=(-2.0, 0.1)),
            seed=args['trial'],
            thresholds=10**np.linspace(-3, 0, 50),
            durations=np.linspace(10, 100, 10),
            noise_levels=np.linspace(0.01, 0.10, 10),
            sample_rates=(0.01,),
            sparsity_search=(2, 10),
        )
    elif args['system'].lower() == 'test':
        results = constraint_benchmark(
            system=Duffing(library_size=10),
            thresholds=(0.1, 0.2),
            seed=args['trial'],
            durations=(20, 60),
            noise_levels=(0.04, 0.08),
            sample_rates=(0.01,),
            sparsity_search=(4, 10),
        )
    else:
        raise ValueError('Not a valid system')

    result_root_dir = os.path.join(os.getcwd().split('sindy_mio')[0], 'sindy_mio', 'results')
    experiment_name = f'{args["name"]}_{args["system"]}'
    experiment_path = os.path.join(result_root_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    result_file = os.path.join(experiment_path, f'trial_{args["trial"]}.csv')
    results.to_csv(result_file, index=False)
