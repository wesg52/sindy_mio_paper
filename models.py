import numpy as np
import pysindy as ps
import copy
from itertools import product
from miosr import *
from evaluation_metrics import *


def fit_and_tune_miosos(feature_library, dif_method, x_train, dt, x_valid,
                        k_span=(1, 5), alphas=(0,)):
    aics = []
    coef_history = []
    params = []
    for k, alpha in product(range(k_span[0], k_span[1] + 1), alphas):
        group_sparsity = tuple(k for _ in range(x_train.shape[1]))
        mio_model = ps.SINDy(
            optimizer=MIOSR(group_sparsity=group_sparsity, alpha=alpha),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        mio_model.fit(x_train, t=dt, quiet=True)
        coef_history.append(mio_model.optimizer.coef_)
        params.append((k, alpha))

        x_dot_valid = mio_model.differentiate(x_valid, dt)
        x_dot_pred = mio_model.predict(x_valid)
        k = (np.abs(mio_model.coefficients()) > 0).sum()

        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=True)
        aics.append(aic)

    min_aic_per_dimension = np.argmin(np.array(aics), axis=0)
    coef_history = np.array(coef_history)
    best_model = coef_history[min_aic_per_dimension, np.arange(len(min_aic_per_dimension))]
    mio_model.optimizer.coef_ = best_model

    best_group_sparsity = tuple(params[i][0] for i in min_aic_per_dimension)
    best_alphas = tuple(params[i][1] for i in min_aic_per_dimension)
    mio_model.group_sparsity = best_group_sparsity
    return mio_model, (('group_sparsity', best_group_sparsity), ('alpha', best_alphas))


def fit_and_tune_stlsq(feature_library, dif_method, x_train, dt, x_valid,
                       thresholds=(0.005, 0.01, 0.05, 0.1, 0.2, 0.5),
                       alphas=(0, 0.05, 1)):
    aics = []
    parameters = list(product(thresholds, alphas))
    for threshold, alpha in parameters:
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        model.fit(x_train, t=dt, quiet=True)

        x_dot_valid = model.differentiate(x_valid, dt)
        x_dot_pred = model.predict(x_valid)
        k = (np.abs(model.coefficients()) > 0).sum()

        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)

    best_t, best_a = parameters[np.argmin(np.array(aics))]
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=best_t, alpha=best_a),
        feature_library=feature_library,
        differentiation_method=dif_method
    )
    model.fit(x_train, t=dt, quiet=True)
    return model, (('threshold', best_t), ('alpha', best_a))


def fit_and_tune_ssr(feature_library, dif_method, library_dim, x_train, dt, x_valid, alphas):
    aics = []
    coef_histories = []
    for alpha in alphas:
        ssr_model = ps.SINDy(
            optimizer=ps.SSR(max_iter=library_dim - 1, alpha=alpha),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        ssr_model.fit(x_train, t=dt, quiet=True)
        for j in range(len(ssr_model.optimizer.history_)):
            ssr_model.optimizer.coef_ = ssr_model.optimizer.history_[j]
            x_dot_valid = ssr_model.differentiate(x_valid, dt)
            x_dot_pred = ssr_model.predict(x_valid)
            k = (np.abs(ssr_model.coefficients()) > 0).sum()

            aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=True)
            aics.append(aic)
        coef_histories.append(np.array(ssr_model.optimizer.history_))
    min_aic_per_dimension = np.argmin(np.array(aics), axis=0)
    coef_history = np.vstack(coef_histories)
    best_model = coef_history[min_aic_per_dimension, np.arange(len(min_aic_per_dimension))]
    ssr_model.optimizer.coef_ = best_model
    best_alphas = np.array(alphas)[min_aic_per_dimension // len(coef_histories[0])]
    return ssr_model, (('alpha', tuple(best_alphas)),)


def fit_and_tune_sr3(feature_library, dif_method, x_train, dt, x_valid, thresholds,
                     nus=(1 / 30, 0.1, 1 / 3, 1, 10 / 3)):
    aics = []
    params = list(product(thresholds, nus))
    for threshold, nu in params:
        model = ps.SINDy(
            optimizer=ps.SR3(threshold=threshold, nu=nu, max_iter=1000),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        try:
            model.fit(x_train, t=dt, quiet=True)

            x_dot_valid = model.differentiate(x_valid, dt)
            x_dot_pred = model.predict(x_valid)
            k = (np.abs(model.coefficients()) > 0).sum()
            aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
            aics.append(aic)
        except ValueError:  # SR3 sometimes fails with overflow
            aics.append(1e20)

    best_model_ix = np.argmin(np.array(aics))
    best_t, best_nu = params[best_model_ix]
    model = ps.SINDy(
        optimizer=ps.SR3(threshold=best_t, nu=best_nu, max_iter=10000),
        feature_library=feature_library,
        differentiation_method=dif_method
    )
    model.fit(x_train, t=dt, quiet=False)
    return model, (('threshold', best_t), ('nu', best_nu))


def fit_and_tune_ensemble_stlsq(feature_library, dif_method, x_train, dt, x_valid,
                                n_ensemble_models=20,
                                thresholds=(0.005, 0.01, 0.05, 0.1, 0.2, 0.5),
                                alphas=(0.01, 0.05)):
    aics = []
    coef_history = []
    params = list(product(thresholds, alphas))
    for threshold, alpha in params:
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=alpha),
            feature_library=feature_library,
            differentiation_method=dif_method
        )
        model.fit(x_train, t=dt, ensemble=True, n_models=n_ensemble_models, quiet=True)
        median_coefs = np.median(model.coef_list, axis=0)
        model.optimizer.coef_ = median_coefs
        coef_history.append(median_coefs)

        x_dot_valid = model.differentiate(x_valid, dt)
        x_dot_pred = model.predict(x_valid)
        k = (np.abs(median_coefs) > 0).sum()

        aic = AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False)
        aics.append(aic)
    model = ps.SINDy(
        optimizer=ps.STLSQ(),
        feature_library=feature_library,
        differentiation_method=dif_method
    )
    model.fit(x_train, t=dt, quiet=True)
    model.optimizer.coef_ = coef_history[np.argmin(np.array(aics))]
    best_t, best_a = params[np.argmin(np.array(aics))]
    return model, (('threshold', best_t), ('alpha', best_a))
