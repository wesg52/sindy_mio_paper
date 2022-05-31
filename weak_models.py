import numpy as np
import pysindy as ps
import copy
from itertools import product
from pysindy.utils import convert_u_dot_integral
from miosr import *
from evaluation_metrics import *


def make_dummy_model(x_train, library_functions, library_function_names, window_length=21, dt=0.002):
    differentiation_method = ps.SmoothedFiniteDifference(
        smoother_kws={'window_length': window_length},
    )
    feature_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        include_bias=True,
    )
    optimizer = ps.STLSQ(threshold=1e10)
    dummy_model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
        differentiation_method=differentiation_method
    )
    dummy_model.fit(x_train, t=dt, quiet=True)
    return dummy_model


def combined_aic_score(x_dot_pred, x_dot_valid, weak_u_dot_pred, weak_u_dot_valid, k, keep_dimensionalized=True):
    aics = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=keep_dimensionalized)
    aics_weak = AIC(weak_u_dot_valid, weak_u_dot_pred, k, keep_dimensionalized=keep_dimensionalized)

    min_aic_per_dimension = np.argmin(np.array(aics), axis=0)
    min_weak_aic_per_dimension = np.argmin(np.array(aics_weak), axis=0)
    combined_aic = np.vstack([min_aic_per_dimension, min_weak_aic_per_dimension])
    min_aic = np.min(combined_aic)
    return min_aic


def fit_and_tune_weak_miosr(cached_features, feature_library, valid_library, x_train, dt, x_valid,
                             k_span=(1, 5), alphas=(0,), use_weak_xdot_for_validation=False):
    if use_weak_xdot_for_validation:
        weak_theta = valid_library.transform(x_valid)
        # Actually weak_u_dot_valid
        x_dot_valid = convert_u_dot_integral(x_valid, valid_library)
    else:
        dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)

    aics = []
    coef_history = []
    params = list(product(range(k_span[0], k_span[1] + 1), alphas))
    for k, alpha in params:
        group_sparsity = tuple(k for _ in range(x_train.shape[1]))
        u_dot_integral, theta, id_lib = cached_features
        mio_model = ps.SINDy(
            optimizer=MIOSR(group_sparsity=group_sparsity,
                            alpha=alpha,
                            normalize_columns=True),
            feature_library=id_lib
        )
        mio_model.fit(theta, x_dot=u_dot_integral, quiet=True)
        coef_history.append(mio_model.optimizer.coef_)

        if use_weak_xdot_for_validation:
            # Actually weak_u_dot_pred
            x_dot_pred = weak_theta @ mio_model.coefficients().T
        else:
            dummy_model.optimizer.coef_ = coef_history[-1]
            x_dot_valid = dummy_model.differentiate(x_valid, dt)
            x_dot_pred = dummy_model.predict(x_valid)

        k = (np.abs(mio_model.coefficients()) > 0).sum()
        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=True)
        aics.append(aic)

    min_aic_per_dimension = np.argmin(np.array(aics), axis=0)
    coef_history = np.array(coef_history)
    best_model = coef_history[min_aic_per_dimension, np.arange(len(min_aic_per_dimension))]
    dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    dummy_model.optimizer.coef_ = best_model
    best_group_sparsity = tuple(params[i][0] for i in min_aic_per_dimension)
    best_alphas = tuple(params[i][1] for i in min_aic_per_dimension)
    return dummy_model, (('group_sparsity', best_group_sparsity), ('alpha', best_alphas))


def fit_and_tune_weak_stlsq(cached_features, feature_library, valid_library, x_train, dt, x_valid,
                            thresholds=(0.005, 0.01, 0.05, 0.1, 0.2, 0.5),
                            alphas=(0, 0.05, 1),
                            use_weak_xdot_for_validation=False):
    if use_weak_xdot_for_validation:
        weak_theta = valid_library.transform(x_valid)
        # Actually weak_u_dot_valid
        x_dot_valid = convert_u_dot_integral(x_valid, valid_library)
    else:
        dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    aics = []
    parameters = list(product(thresholds, alphas))
    for threshold, alpha in parameters:
        u_dot_integral, theta, id_lib = cached_features
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True),
            feature_library=id_lib
        )
        model.fit(theta, x_dot=u_dot_integral, quiet=True)

        if use_weak_xdot_for_validation:
            # Actually weak_u_dot_pred
            x_dot_pred = weak_theta @ model.coefficients().T
        else:
            dummy_model.optimizer.coef_ = model.optimizer.coef_
            x_dot_valid = dummy_model.differentiate(x_valid, dt)
            x_dot_pred = dummy_model.predict(x_valid)

        k = (np.abs(model.coefficients()) > 0).sum()
        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)

    best_t, best_a = parameters[np.argmin(np.array(aics))]
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=best_t, alpha=best_a, normalize_columns=True),
        feature_library=feature_library
    )
    model.fit(x_train, t=dt, quiet=True)
    dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    dummy_model.optimizer.coef_ = model.coefficients()
    return dummy_model, (('threshold', best_t), ('alpha', best_a))


def fit_and_tune_weak_ssr(cached_features, feature_library, valid_library, x_train, dt, x_valid, alphas,
                          use_weak_xdot_for_validation=False):
    if use_weak_xdot_for_validation:
        weak_theta = valid_library.transform(x_valid)
        # Actually weak_u_dot_valid
        x_dot_valid = convert_u_dot_integral(x_valid, valid_library)
    else:
        dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    aics = []
    coef_histories = []
    for alpha in alphas:
        u_dot_integral, theta, id_lib = cached_features
        ssr_model = ps.SINDy(
            optimizer=ps.SSR(max_iter=theta.shape[1] - 1, alpha=alpha, normalize_columns=True),
            feature_library=id_lib,
        )
        ssr_model.fit(theta, x_dot=u_dot_integral, quiet=True)
        for j in range(len(ssr_model.optimizer.history_)):
            ssr_model.optimizer.coef_ = ssr_model.optimizer.history_[j]
            if use_weak_xdot_for_validation:
                # Actually weak_u_dot_pred
                x_dot_pred = weak_theta @ ssr_model.coefficients().T
            else:
                dummy_model.optimizer.coef_ = ssr_model.optimizer.coef_
                x_dot_valid = dummy_model.differentiate(x_valid, dt)
                x_dot_pred = dummy_model.predict(x_valid)
            k = (np.abs(ssr_model.coefficients()) > 0).sum()
            aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=True)
            aics.append(aic)
        coef_histories.append(np.array(ssr_model.optimizer.history_))
    min_aic_per_dimension = np.argmin(np.array(aics), axis=0)
    coef_history = np.vstack(coef_histories)
    best_model = coef_history[min_aic_per_dimension, np.arange(len(min_aic_per_dimension))]
    best_alphas = np.array(alphas)[min_aic_per_dimension // len(coef_histories[0])]

    dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    dummy_model.optimizer.coef_ = best_model
    return dummy_model, (('alpha', tuple(best_alphas)),)


def fit_and_tune_weak_sr3(cached_features, feature_library, valid_library, x_train, dt, x_valid, thresholds,
                          nus=(1 / 30, 0.1, 1 / 3, 1, 10 / 3),
                          use_weak_xdot_for_validation=False):
    if use_weak_xdot_for_validation:
        weak_theta = valid_library.transform(x_valid)
        # Actually weak_u_dot_valid
        x_dot_valid = convert_u_dot_integral(x_valid, valid_library)
    else:
        dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    aics = []
    params = list(product(thresholds, nus))
    u_dot_integral, theta, id_lib = cached_features
    for threshold, nu in params:
        sr3_model = ps.SINDy(
            optimizer=ps.SR3(threshold=threshold, nu=nu, max_iter=10000, normalize_columns=True),
            feature_library=id_lib
        )
        sr3_model.fit(theta, x_dot=u_dot_integral, quiet=True)
        if use_weak_xdot_for_validation:
            # Actually weak_u_dot_pred
            x_dot_pred = weak_theta @ sr3_model.coefficients().T
        else:
            dummy_model.optimizer.coef_ = sr3_model.optimizer.coef_
            x_dot_valid = dummy_model.differentiate(x_valid, dt)
            x_dot_pred = dummy_model.predict(x_valid)

        k = (np.abs(sr3_model.coefficients()) > 0).sum()
        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)

    best_model_ix = np.argmin(np.array(aics))
    best_t, best_nu = params[best_model_ix]
    model = ps.SINDy(
        optimizer=ps.SR3(threshold=best_t, nu=best_nu, max_iter=10000, normalize_columns=True),
        feature_library=id_lib
    )
    model.fit(theta, x_dot=u_dot_integral, quiet=False)
    dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    dummy_model.optimizer.coef_ = model.coefficients()
    return dummy_model, (('threshold', best_t), ('nu', best_nu))


def fit_and_tune_weak_ensemble_stlsq(cached_features, feature_library, valid_library, x_train, dt, x_valid,
                                     thresholds=(0.005, 0.01, 0.05, 0.1, 0.2, 0.5),
                                     alphas=(0.05,),
                                     n_ensemble_models=20,
                                     use_weak_xdot_for_validation=False):
    if use_weak_xdot_for_validation:
        weak_theta = valid_library.transform(x_valid)
        # Actually weak_u_dot_valid
        x_dot_valid = convert_u_dot_integral(x_valid, valid_library)
    else:
        dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    aics = []
    coef_history = []
    params = list(product(thresholds, alphas))
    for threshold, alpha in params:
        u_dot_integral, theta, id_lib = cached_features
        ensemble_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=True),
            feature_library=id_lib
        )
        ensemble_model.fit(theta, x_dot=u_dot_integral, ensemble=True, n_models=n_ensemble_models,
                  quiet=True)
        median_coefs = np.median(ensemble_model.coef_list, axis=0)
        ensemble_model.optimizer.coef_ = median_coefs

        coef_history.append(median_coefs)

        if use_weak_xdot_for_validation:
            # Actually weak_u_dot_pred
            x_dot_pred = weak_theta @ ensemble_model.coefficients().T
        else:
            dummy_model.optimizer.coef_ = median_coefs
            x_dot_valid = dummy_model.differentiate(x_valid, dt)
            x_dot_pred = dummy_model.predict(x_valid)

        k = (np.abs(median_coefs) > 0).sum()
        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)

    dummy_model = make_dummy_model(x_train, feature_library.functions, feature_library.function_names, dt=dt)
    dummy_model.optimizer.coef_ = coef_history[np.argmin(np.array(aics))]
    best_t, best_a = params[np.argmin(np.array(aics))]
    return dummy_model, (('threshold', best_t), ('alpha', best_a))
