import os
import pandas as pd
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

metric_name_map = {
    'support_accuracy': 'True Positivity Rate',
    'coef_l2': '$L_2$ Coefficient Error',
    'test_rmse': '$\dot{X}$ Test RMSE',
    'constraint_violation': 'Constraint Violation'
}


def load_results(experiments, clip_prefix=True):
    results = {}
    for experiment in experiments:
        experiment_path = os.path.join(os.getcwd(), 'results', experiment)
        dfs = []
        for f in os.listdir(experiment_path):
            trial = pd.read_csv(os.path.join(experiment_path, f))
            dfs.append(trial)
        fdf = pd.concat(dfs)
        key = experiment.split('_')[-1] if clip_prefix else experiment
        results[key] = fdf
    return results


def multi_system_differential_benchmark(result_dict,
                                        methods=('MIOSR', 'STLSQ', 'SSR', 'SR3', 'E-STLSQ',),
                                        save_name='differential.png'):
    metrics = ['support_accuracy', 'coef_l2', 'test_rmse']
    system_name_map = {'lorenz': 'Lorenz', 'hopf': 'Hopf', 'mhd': 'MHD'}
    fig, axs = plt.subplots(len(metrics), len(result_dict),
                            figsize=(16, 12), sharex='col')
    for exp_ix, (experiment, results) in enumerate(result_dict.items()):
        for m in metrics[1:]:
            results.loc[:, f'log_{m}'] = np.log10(results[m])
        for method in methods:
            group = results.query('Method == @method')
            for metric_ix, metric in enumerate(metrics):
                ax = axs[metric_ix, exp_ix]
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                if metric == 'support_accuracy':
                    metric_summary = group.groupby('Duration')[metric].describe()
                    x = metric_summary.index.values
                    stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                    y = metric_summary['mean'].values
                    y_std = metric_summary['std'].values
                    ax.errorbar(x, y, yerr=y_std * stderr_multiplier, label=method, capsize=3)
                else:  # else log plot
                    metric_summary = group.groupby('Duration')[f'log_{metric}'].describe()
                    x = metric_summary.index.values
                    stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                    log_y = metric_summary['mean'].values
                    y = 10 ** log_y
                    log_y_std = metric_summary['std'].values
                    yerr_ub = 10 ** (log_y + (log_y_std * stderr_multiplier)) - y
                    yerr_lb = y - 10 ** (log_y - (log_y_std * stderr_multiplier))
                    ax.errorbar(x, y, yerr=np.vstack([yerr_lb, yerr_ub]), label=method, capsize=3)

                ax.set_xscale('log')
                if metric[:7] == 'support':
                    ax.set_ylim([-0.05, 1.05])
                if metric[:4] == 'coef':
                    ax.set_yscale('log')
                if metric[-4:] == 'rmse':
                    ax.set_yscale('log')
                if metric_ix == 2:
                    ax.set_xlabel('Training Trajectory Length (s)')
                if metric_ix == 0:
                    ax.set_title(system_name_map.get(experiment, experiment))
                if exp_ix == 0:
                    ax.set_ylabel(metric_name_map.get(metric, metric))
    plt.subplots_adjust(hspace=0.05)
    ax.legend()
    fig.savefig(os.path.join('plots', save_name),
                bbox_inches='tight')


def multi_system_integral_benchmark(result_dict,
                                    methods=('MIOSR', 'STLSQ', 'SSR', 'SR3', 'E-STLSQ'),
                                    save_name='integral.png'):
    metrics = ['support_accuracy', 'coef_l2', 'test_rmse']
    system_name_map = {'vanderpol': 'Van der Pol', 'lotka': 'Lotka', 'rossler': 'Rossler'}
    fig, axs = plt.subplots(len(metrics), len(result_dict),
                            figsize=(16, 12), sharex='col')
    for exp_ix, (experiment, results) in enumerate(result_dict.items()):
        for m in metrics[1:]:
            results.loc[:, f'log_{m}'] = np.log10(results[m])
        for method in methods:
            group = results.query('Method == @method')
            for metric_ix, metric in enumerate(metrics):
                ax = axs[metric_ix, exp_ix]
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if metric == 'support_accuracy':
                    metric_summary = group.groupby('Noise')[metric].describe()
                    x = metric_summary.index.values * 100
                    stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                    y = metric_summary['mean'].values
                    y_std = metric_summary['std'].values
                    ax.errorbar(x, y, yerr=y_std * stderr_multiplier, label=method, capsize=3)
                else:  # else log plot
                    metric_summary = group.groupby('Noise')[f'log_{metric}'].describe()
                    x = metric_summary.index.values * 100
                    stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                    log_y = metric_summary['mean'].values
                    y = 10 ** log_y
                    log_y_std = metric_summary['std'].values
                    yerr_ub = 10 ** (log_y + (log_y_std * stderr_multiplier)) - y
                    yerr_lb = y - 10 ** (log_y - (log_y_std * stderr_multiplier))
                    ax.errorbar(x, y, yerr=np.vstack([yerr_lb, yerr_ub]), label=method, capsize=3)
                ax.set_xscale('log')
                if metric[:7] == 'support':
                    ax.set_ylim([-0.05, 1.05])
                if metric[:4] == 'coef':
                    ax.set_yscale('log')
                if metric[-4:] == 'rmse':
                    ax.set_yscale('log')
                if metric_ix == 2:
                    ax.set_xlabel('Noise %')
                if metric_ix == 0:
                    ax.set_title(system_name_map.get(experiment, experiment))
                if exp_ix == 0:
                    ax.set_ylabel(metric_name_map.get(metric, metric))
    plt.subplots_adjust(hspace=0.05)
    ax.legend()
    fig.savefig(os.path.join('plots', save_name),
                bbox_inches='tight')


def multi_system_pde_benchmark(result_dict,
                               methods,
                               save_name='pde.png'):
    metrics = ['support_accuracy', 'coef_l2']
    system_name_map = {'ks': 'Kuramoto-Sivashinsky', 'redif': 'Reaction Diffusion'}
    fig, axs = plt.subplots(len(metrics), len(result_dict),
                            figsize=(14, 8), sharex='col')
    for exp_ix, (experiment, results) in enumerate(result_dict.items()):
        for m in metrics[1:]:
            results.loc[:, f'log_{m}'] = np.log10(results[m])
        for method in methods:
            group = results.query('Method == @method')
            for metric_ix, metric in enumerate(metrics):
                ax = axs[metric_ix, exp_ix]
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if metric == 'support_accuracy':
                    metric_summary = group.groupby('Noise')[metric].describe()
                    x = metric_summary.index.values * 100
                    stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                    y = metric_summary['mean'].values
                    y_std = metric_summary['std'].values
                    ax.errorbar(x, y, yerr=y_std * stderr_multiplier, label=method, capsize=3)
                else:  # else log plot
                    metric_summary = group.groupby('Noise')[f'log_{metric}'].describe()
                    x = metric_summary.index.values * 100
                    stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                    log_y = metric_summary['mean'].values
                    y = 10 ** log_y
                    log_y_std = metric_summary['std'].values
                    yerr_ub = 10 ** (log_y + (log_y_std * stderr_multiplier)) - y
                    yerr_lb = y - 10 ** (log_y - (log_y_std * stderr_multiplier))
                    ax.errorbar(x, y, yerr=np.vstack([yerr_lb, yerr_ub]), label=method, capsize=3)
                if metric[:7] == 'support':
                    ax.set_ylim(bottom=0.19, top=1.05)
                if metric[:4] == 'coef':
                    ax.set_yscale('log')
                if metric[-4:] == 'rmse':
                    ax.set_yscale('log')
                if metric_ix == 1:
                    ax.set_xlabel('Noise %')
                if metric_ix == 0:
                    ax.set_title(system_name_map.get(experiment, experiment))
                if exp_ix == 0:
                    ax.set_ylabel(metric_name_map.get(metric, metric))
    plt.subplots_adjust(hspace=0.05)
    ax.legend()
    if save_name:
        fig.savefig(os.path.join('plots', save_name),
                    bbox_inches='tight')


def runtime_plots(result_dict, prefix,
                  methods=('MIOSR', 'STLSQ', 'SSR', 'SR3', 'E-STLSQ'),
                  save_name='runtime.png'):
    systems = ['lorenz', 'hopf', 'mhd']
    name_map = {'lorenz': 'Lorenz', 'hopf': 'Hopf', 'mhd': 'MHD'}
    orders = ['5', '3']
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex='col')
    for col_ix, system in enumerate(systems):
        for row_ix, order in enumerate(orders):
            key = f'{prefix}_{system}_{order}'
            rdf = result_dict[key]
            dim = rdf.Dim.unique()[0]
            rdf['log_runtime'] = np.log10(rdf.runtime.values)
            ax = axs[row_ix, col_ix]
            for method_ix, method in enumerate(methods):
                group = rdf.query('Method == @method')
                metric_summary = group.groupby('Duration')['log_runtime'].describe()
                x = metric_summary.index.values
                stderr_multiplier = 1.96 / metric_summary['count'].values ** 0.5
                log_y = metric_summary['mean'].values
                y = 10 ** log_y
                log_y_std = metric_summary['std'].values
                yerr_ub = 10 ** (log_y + (log_y_std * stderr_multiplier)) - y
                yerr_lb = y - 10 ** (log_y - (log_y_std * stderr_multiplier))
                ax.errorbar(x, y, yerr=np.vstack([yerr_lb, yerr_ub]), label=method, capsize=3)

                ax.set_xscale('log')
                ax.set_yscale('log')
                if row_ix == 1:
                    ax.set_xlabel('Training Trajectory Length (s)')
                if col_ix == 0:
                    ax.set_ylabel('Runtime (s)')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            ax.set_title(f'{name_map[system]} (Order={order}, D={dim})')
    ax.legend(ncol=6, bbox_to_anchor=(-0.7, -0.38), loc='lower center')
    fig.savefig(os.path.join('plots', save_name),
                bbox_inches='tight')


def noise_duration_grid_plot(  # Constraint experiment plot
        result_df,
        methods=('MIOSR', 'Con-MIOSR', 'SR3', 'Con-SR3'),
        metrics=('support_accuracy', 'coef_l2', 'test_rmse', 'constraint_violation'),
        htitle_offset=0.5,
        save_name=''
):
    gdf = result_df.groupby(['Method', 'Duration', 'Noise']).mean().reset_index()
    noise = np.round(sorted(gdf.Noise.unique() * 100)).astype(int)
    durations = np.array(sorted(gdf.Duration.unique())).astype(int)

    fig, axs = plt.subplots(len(metrics), len(methods),
                            figsize=(3.5 * len(metrics), 3.5 * len(methods)),
                            sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0.02, wspace=-0.08, top=0.75)

    for method_ix, method in enumerate(methods):
        method_df = gdf.query("Method == @method")
        for metric_ix, metric in enumerate(metrics):
            ax = axs[metric_ix, method_ix]
            subdf = method_df[['Duration', 'Noise', metric]].set_index(['Duration', 'Noise']).unstack()
            if metric_ix == 0:
                im = ax.matshow(subdf, vmin=max(np.min(gdf[metric]), 0.59), vmax=np.max(gdf[metric]))
            else:
                im = ax.matshow(subdf,
                                norm=colors.LogNorm(vmin=max(np.min(gdf[metric]), 1e-6),
                                                    vmax=np.max(gdf[metric])))

            if method_ix == len(methods) - 1:
                fig.colorbar(im, ax=axs[metric_ix, :], shrink=1, pad=0.01, aspect=8)
            if method_ix == 0:
                ax.set_ylabel('Duration (s)')
                ax.annotate(metric_name_map.get(metric, metric),
                            xy=(-htitle_offset, 0.5),
                            xycoords='axes fraction', rotation=90, va='center',
                            size=16)
            if metric_ix == len(metrics) - 1:
                ax.set_xlabel('Noise %')
                ax.tick_params(labelbottom=True, labeltop=False)
            if metric_ix == 0:
                ax.set_title(method, fontsize=16, y=1.05)
            ax.set_xticks(np.arange(len(noise)), noise)
            ax.set_yticks(np.arange(len(durations)), durations)

    if save_name:
        fig.savefig(os.path.join('plots', save_name),
                    bbox_inches='tight')


if __name__ == '__main__':
    # Full lib dif
    result_dict = load_results([
        'final_draft_dif_lorenz',
        'final_draft_dif_hopf',
        'final_draft_dif_mhd'])
    multi_system_differential_benchmark(result_dict)

    # Small lib dif
    result_dict = load_results([
        'final_draft_dif_small_lib_lorenz',
        'final_draft_dif_small_lib_hopf',
        'final_draft_dif_small_lib_mhd'
    ])
    multi_system_differential_benchmark(result_dict, save_name='differential_small_lib.png')

    # Weak form
    result_dict = load_results([
        'weak_final_draft_vanderpol',
        'weak_final_draft_lotka',
        'weak_final_draft_rossler'
    ], clip_prefix=True)
    multi_system_integral_benchmark(result_dict, save_name='integral.png')

    # Runtime
    result_dict = load_results([
        'runtime_final_hopf_3', 'runtime_final_hopf_5',
        'runtime_final_lorenz_3', 'runtime_final_lorenz_5',
        'runtime_final_mhd_3', 'runtime_final_mhd_5'
    ], clip_prefix=False)
    local_mio_result_dict = load_results([
        'runtime_local_mio_hopf_3', 'runtime_local_mio_hopf_5',
        'runtime_local_mio_lorenz_3', 'runtime_local_mio_lorenz_5',
        'runtime_local_mio_mhd_3', 'runtime_local_mio_mhd_5'
    ], clip_prefix=False)
    for k, sysdf in local_mio_result_dict.items():
        system = '_'.join(k.split('_')[-2:])
        miodf = sysdf.query('Method == "MIOSOS"')
        miodf = miodf.replace({'MIOSOS': 'MIOSR-M1MAX'})
        result_dict['runtime_final_' + system] = pd.concat([result_dict['runtime_final_' + system], miodf])
    runtime_plots(result_dict, methods=('MIOSR', 'STLSQ', 'SSR', 'SR3', 'E-STLSQ', 'MIOSR-M1MAX'),
                  prefix='runtime_final', save_name='runtime.png')

    # Runtime local
    result_dict = load_results([
        'runtime_local_hopf_3', 'runtime_local_hopf_5',
        'runtime_local_lorenz_3', 'runtime_local_lorenz_5',
        'runtime_local_mhd_3', 'runtime_local_mhd_5'
    ], clip_prefix=False)
    local_mio_result_dict = load_results([
        'runtime_local_mio_hopf_3', 'runtime_local_mio_hopf_5',
        'runtime_local_mio_lorenz_3', 'runtime_local_mio_lorenz_5',
        'runtime_local_mio_mhd_3', 'runtime_local_mio_mhd_5'
    ], clip_prefix=False)
    for k, sysdf in local_mio_result_dict.items():
        system = '_'.join(k.split('_')[-2:])
        miodf = sysdf.query('Method == "MIOSOS"')
        miodf = miodf.replace({'MIOSOS': 'MIOSR'})
        result_dict['runtime_local_' + system] = pd.concat([result_dict['runtime_local_' + system], miodf])

    # Constraint
    result_dict = load_results(['constraint_final_duffing'], clip_prefix=True)
    noise_duration_grid_plot(result_dict['duffing'], save_name='constraint_final.png')

    # PDEs
    result_dict = load_results(['pde_final_ks', 'pde_final_redif'], clip_prefix=True)
    methods = ('MIOSR', 'STLSQ', 'SSR', 'SR3', 'E-STLSQ')
    multi_system_pde_benchmark(result_dict, methods, save_name='pde_final.png')



