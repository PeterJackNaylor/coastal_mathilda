import os
# import sys 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'INR4Torch')))
import argparse
import pinns
import pandas as pd
import torch
import torch.nn as nn
from math import ceil
import numpy as np
from functools import partial
import matplotlib.pylab as plt
import matplotlib.cm as cm
from pinns.models import INR
from datetime import datetime
from dataloader import return_dataset, load_data, load_data_faster, load_eval_data_faster
from pde_model import Surface
from tqdm import tqdm 
from shutil import copy 
import optuna
import gc
from pc_utils import *
from plot_utils import *
from temp_encoding import *
from evaluations import *


def setup_uniform_grid(pc, step):
    xmin, ymin, xmax, ymax = pc[:, 1].min(), pc[:, 2].min(), pc[:, 1].max(), pc[:, 2].max()
    xx_grid = np.arange(xmin, xmax, step)
    yy_grid = np.arange(ymin, ymax, step * 2)
    xx, yy = np.meshgrid(
        xx_grid,
        yy_grid,
    )
    xx = xx.astype(float)
    yy = yy.astype(float)
    samples = np.vstack([xx.ravel(), yy.ravel()]).T
    n, p = xx_grid.shape[0], yy_grid.shape[0]
    return samples, n, p


def parser_f():
    parser = argparse.ArgumentParser(
        description="Estimating surface with INR",
    )
    parser.add_argument("--data", type=str, help="Data path (npy)")
    parser.add_argument("--name", type=str, help="Name given to saved files")
    parser.add_argument(
        "--yaml_file",
        type=str,
        default="mathilda.yml",
        help="Configuration yaml file for the INR hyper-parameters",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="keyword: seasonal, default",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        help="File with test indexes, only to use with specific datasets",
    )
    args = parser.parse_args()
    # args.name = f"outputs/{args.name}"
    return args


def setup_hp(
    yaml_params,
    data,
    name,
):
    try:
        os.mkdir(name)
    except:
        pass

    copy(yaml_params, f"{name}/used_config.yml")
    model_hp = pinns.read_yaml(yaml_params)
    gpu = torch.cuda.is_available()
    # device = "cuda" if gpu else "cpu"

    n = int(data.shape[0] * model_hp.train_fraction)
    bs = model_hp.losses["mse"]["bs"]
    model_hp.max_iters = ceil(n // bs) * model_hp.epochs
    model_hp.test_frequency = ceil(n // bs) * model_hp.test_epochs
    model_hp.learning_rate_decay["step"] = (
        ceil(n // bs) * model_hp.learning_rate_decay["epoch"]
    )
    model_hp.cosine_anealing["step"] = ceil(n // bs) * model_hp.cosine_anealing["epoch"]
    model_hp.gpu = gpu
    model_hp.verbose = True
    # model_hp.model["name"] = model
    # model_hp.model["scale"] = scale
    # model_hp.pth_name = f"{name}.pth"
    # model_hp.npz_name = f"{name}.npz"
    return model_hp


def save_results(variables, name, suffix=""):
    results = pd.DataFrame()
    # variables = [t_var, mae_train, mae_test, mae_all, rmse_train, rmse_test, rmse_all]
    names = [
        "MAE (Train)",
        "MAE (Validation)",
        "MAE (Train + V)",
        "RMSE (Train)",
        "RMSE (Validation)",
        "RMSE (Train + V)",
        "MAE (Test)",
        "RMSE (Test)",
    ]
    for v, n in zip(variables, names):
        results.loc[name.split("/")[-1], n] = v
    results.to_csv(f"{name}/results{suffix}.csv")


def plot_NN(NN, model_hp, name):
    try:
        os.mkdir(f"{name}/plots")
    except:
        pass
    n = len(NN.test_scores)
    f = model_hp.test_frequency
    plt.plot(list(range(1 * f, (n + 1) * f, f)), NN.test_scores)
    plt.savefig(f"{name}/plots/test_scores.png")
    plt.close()

    for k in NN.loss_values.keys():
        try:
            loss_k = NN.loss_values[k]
            plt.plot(loss_k)
            plt.savefig(f"{name}/plots/{k}.png")
            plt.close()
        except:
            print(f"Couldn't plot {k}")
    try:
        plt.plot([np.log(lr) / np.log(10) for lr in NN.lr_list])
        plt.savefig(f"{name}/plots/LR.png")
        plt.close()
    except:
        print("Coulnd't plot LR")
    try:
        if model_hp.relobralo["status"]:
            f = model_hp.relobralo["step"]
        elif model_hp.self_adapting_loss_balancing["status"]:
            f = model_hp.self_adapting_loss_balancing["step"]

        for k in NN.lambdas_scalar.keys():
            n = len(NN.lambdas_scalar[k])
            plt.plot(list(range(0, n * f, f)), NN.lambdas_scalar[k], label=k)
        plt.legend()
        plt.savefig(f"{name}/plots/lambdas_scalar.png")
        plt.close()
    except:
        print(f"{name}/Couldn't plot lambdas_scalar")

    for key in NN.temporal_weights.keys():
        try:
            f = model_hp.temporal_causality["step"]
            t_weights = torch.column_stack(NN.temporal_weights[key])
            x_axis = t_weights.shape[1]  # because we will remove the first one
            x_axis = list(range(0, x_axis * f, f))
            if model_hp.gpu:
                t_weights = t_weights.cpu()
            color = cm.hsv(np.linspace(0, 1, t_weights.shape[0]))
            for k in range(t_weights.shape[0]):
                plt.plot(x_axis, t_weights[k], label=f"w_{k}", color=color[k])
            plt.legend()
            plt.savefig(f"plots/w_temp_{key}_weights.png")
            plt.close()
        except:
            print(f"{name}/Couldn't plot t_weights for {key}")


def split_test(data, index):
    if index is None:
        return data, data[:1000].copy(), None
    idx = (index == 2).squeeze()
    data_test = data[idx, :]
    data_train = data[~idx, :]
    return data_train, data_test, index[~idx]


def sample_hp(hp, trial):
    # hp.model["epochs"] = trial.suggest_int("epochs", 50, 400, log=True)
    hp.model["epochs"] = trial.suggest_categorical('epochs', [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400])
    hp.model["mapping_size"] = trial.suggest_categorical('mapping_size', [128,256,512])
    hp.lr = trial.suggest_float(
                    "lr",
                    1e-4,
                    1e-2,
                    log=True,
                )
    hp.model["scale"] = trial.suggest_float("scale", 1e-2, 5, log=True)
    hp.model["scale_time"] = trial.suggest_float("scale_time", 1e-2, 5, log=True)
    hp.losses["gradient_lat"]["lambda"] = trial.suggest_float(
                    "lambda_lat",
                    1e-2,
                    10,
                    log=True,
                )
    hp.losses["gradient_lon"]["lambda"] = trial.suggest_float(
                    "lambda_lon",
                    1e-2,
                    10,
                    log=True,
                )
    return hp


def free_gpu(pinns_object):
    pinns_object.model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def objective_optuna(trial, model_hp, data_fn, name):
    try:
        os.mkdir(f"{name}/multiple")
    except:
        pass
    model_hp = sample_hp(model_hp, trial)
    model_hp.pth_name = f"{name}/multiple/optuna_{trial.number}.pth"
    model_hp.npz_name = f"{name}/multiple/optuna_{trial.number}.npz"

    NN, model_hp = pinns.train(
        model_hp, Surface, data_fn, INR, trial=trial, gpu=model_hp.gpu
    )
    try:
        scores = min(NN.test_scores)
        try:
            name = f"{name}/multiple/optuna_{trial.number}"
            os.mkdir(name)
        except:
            pass
        plot_NN(NN, model_hp, name)
    except:
        scores = np.finfo(float).max

    free_gpu(NN)
    return scores


def single_run(
    yaml_params,
    data,
    idx,
    name,
    encoding,
):
    model_hp = setup_hp(
        yaml_params,
        data,
        name,
    )

    return_dataset_fn = partial(return_dataset, data=data, index=idx, encoding=encoding)
    model_hp.pth_name = f"{name}/{name}.pth"
    model_hp.npz_name = f"{name}/{name}.npz"
    NN, model_hp = pinns.train(
        model_hp, Surface, return_dataset_fn, INR, gpu=model_hp.gpu
    )
    return NN, model_hp


def get_n_best_trials(study):
    """
    This function returns the sorted best trials from a study in Optuna.

    Args:
        study: The Optuna study object.

    Returns:
        A list containing the trials sorted by objective value (descending).
    """
    all_trials = study.trials  # Get all trials
    sorted_trials = sorted(all_trials, key=lambda trial: trial.value, reverse=True)
    out = [[el.number, el.values[0]] for el in sorted_trials]
    return out  # Return the top n trials


def load_model(model_hp, weights, npz_path, data, index, encoding):
    npz = np.load(npz_path, allow_pickle=True)
    # import pdb; pdb.set_trace()
    model_hp.input_size = int(npz["input_size"])
    model_hp.output_size = int(npz["output_size"])
    model_hp.model["mapping_size"] = int(npz["model"].item()["mapping_size"])
    model_hp.nv_samples = [tuple(el) for el in tuple(npz["nv_samples"])]
    model_hp.nv_targets = [tuple(el) for el in tuple(npz["nv_targets"])]
    model_hp.model["hidden_nlayers"] = npz["model"].item()["hidden_nlayers"]
    if model_hp.model["name"] == "KAN":
        model_hp.model["hidden_width"] = npz["model"].item()["hidden_width"]
    model = INR(
        model_hp.model["name"],
        model_hp.input_size,
        output_size=model_hp.output_size,
        hp=model_hp,
    )
    if model_hp.gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(weights, map_location=model_hp.device))

    train, test = return_dataset(model_hp, data, model_hp.gpu, index, encoding)
    NN = Surface(train, test, model, model_hp, model_hp.gpu)
    return NN


def plot_optuna(study, name):
    import kaleido
    try:
        os.mkdir(f"{name}/optuna")
        # os.mkdir("optuna")
    except:
        pass
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image(f"{name}/optuna/{name}" + "_inter_optuna.png")
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"{name}/optuna/{name}" + "_searchplane.png")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{name}/optuna/{name}" + "_important_params.png")


def main():
    opt = parser_f()

    keyword = opt.keyword
    if "encoding" in keyword:
        encoding = True
    else:
        encoding = False
    # data = load_data("data/4dinr_synthetic_data.h5")
    data, index = load_data_faster(keyword)

    data_train, data_test, idx = split_test(data, index)
    model_hp = setup_hp(
        opt.yaml_file,
        data_train,
        opt.name,
    )
    train_dataloader, val_dataloader = return_dataset(model_hp, data, model_hp.gpu, index, encoding)
    def dataset_fn(hp, gpu):
        return train_dataloader, val_dataloader
    objective = partial(objective_optuna, model_hp=model_hp, data_fn=dataset_fn, name=opt.name)

    study = optuna.create_study(
        study_name=opt.name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=model_hp.optuna["trials"])
    best_trial = study.best_trial
    print(best_trial)
    best_params = best_trial.params
    pd.DataFrame([{**best_params, 'value': best_trial.value}]). to_csv(
        f"{opt.name}__best_params.csv"
    )
    scores_id = get_n_best_trials(study)
    id_trial = scores_id[-1][0]
    print("Best trial is trial ", id_trial)
    npz = f"{opt.name}/multiple/optuna_{id_trial}.npz"
    weights = f"{opt.name}/multiple/optuna_{id_trial}.pth"
    pd.DataFrame(scores_id, columns=["number", "value"]).to_csv(
        f"{opt.name}__trial_scores.csv"
    )
    model_hp.device = "cuda" if model_hp.gpu else "cpu"
    NN = load_model(model_hp, weights, npz, data, index, encoding)
    # time_preds = plot(data, NN, opt.name, 0, True)  # 0 is trial
    metrics = evaluation(NN, opt.name, encoding)
    metrics_test = evaluation_test(NN, data_test, opt.name, encoding)
    change_data, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean = load_eval_data_faster(keyword)
    evaluation_with_change(NN, change_data, opt.name, encoding)
    evaluation_timeseries(NN, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean, opt.name, encoding)
    # import pdb; pdb.set_trace()
    save_results(metrics + metrics_test, opt.name)
    plot_optuna(study, opt.name)


def main_sr():
    opt = parser_f()

    keyword = opt.keyword
    if "encoding" in keyword:
        encoding = True
    else:
        encoding = False
    data, index = load_data_faster(keyword)

    data_train, data_test, idx = split_test(data, index)
    model_hp = setup_hp(
        opt.yaml_file,
        data_train,
        opt.name,
    )
    
    NN, model_hp = single_run(opt.yaml_file, data, index, opt.name, encoding)
    plot_NN(NN, model_hp, opt.name)
    metrics = evaluation(NN, opt.name, encoding)
    metrics_test = evaluation_test(NN, data_test, opt.name, encoding, suffix="test_last")
    change_data, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean = load_eval_data_faster(keyword)
    evaluation_with_change(NN, change_data, opt.name, encoding, suffix="test_last")
    evaluation_timeseries(NN, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean, opt.name, encoding)
    # import pdb; pdb.set_trace()
    save_results(metrics + metrics_test, opt.name, suffix="last")

    npz = f"{opt.name}/{opt.name}.npz"
    weights = f"{opt.name}/{opt.name}.pth"
    model_hp.device = "cuda" if model_hp.gpu else "cpu"
    NN = load_model(model_hp, weights, npz, data, index, encoding)
    metrics = evaluation(NN, opt.name, encoding)
    metrics_test = evaluation_test(NN, data_test, opt.name, encoding, suffix="test_best")
    evaluation_with_change(NN, change_data, opt.name, encoding, suffix="test_best")
    evaluation_timeseries(NN, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean, opt.name, encoding)
    save_results(metrics + metrics_test, opt.name, suffix="best")


def eval_main():
    opt = parser_f()
    keyword = opt.keyword
    if "encoding" in keyword:
        encoding = True
    else:
        encoding = False
    data, index = load_data_faster(keyword)

    data_train, data_test, idx = split_test(data, index)
    model_hp = setup_hp(
        opt.yaml_file,
        data_train,
        opt.name,
    )

    npz = f"{opt.name}/{opt.name}.npz"
    weights = f"{opt.name}/{opt.name}.pth"
    # OR
    # npz = f"{opt.name}/multiple/optuna_{167}.npz"
    # weights = f"{opt.name}/multiple/optuna_{167}.pth"
    
    model_hp.device = "cuda" if model_hp.gpu else "cpu"
    NN = load_model(model_hp, weights, npz, data, index, encoding)
    # metrics = evaluation(NN, opt.name, encoding)
    # metrics_test = evaluation_test(NN, data_test, opt.name, encoding)
    change_data, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean = load_eval_data_faster(keyword)
    # change_data, ts_pts, ts_gt, uncert_data, uncert_t = load_eval_data_faster(keyword)
    # evaluation_with_change(NN, change_data, opt.name, encoding)
    # evaluation_timeseries(NN, ts_pts, ts_gt, uncert_data, uncert_t, opt.name, encoding)
    evaluation_timeseries(NN, ts_pts, ts_gt, uncert_data, uncert_t, zmean, zstd, tmean, opt.name, encoding)
    # 
    # save_results(metrics + metrics_test, opt.name)


if __name__ == "__main__":
    # main()
    # main_sr()
    eval_main()