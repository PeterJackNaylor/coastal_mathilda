import os
import sys 
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
# from dataloader import return_dataset
# from model_pde import IceSheet
# from utils import grid_on_polygon, predict, inverse_time, load_geojson
from pinns.models import INR
from datetime import datetime
from dataloader import return_dataset
from pde_model import Surface
from datetime import datetime, timedelta
from tqdm import tqdm 
from shutil import copy 
import optuna
import gc


def predict(array, model, attribute="model"):
    n_data = array.shape[0]
    verbose = model.hp.verbose
    bs = model.hp.losses["mse"]["bs"]
    batch_idx = torch.arange(0, n_data, dtype=int, device=model.device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_
    preds = []
    model_function = getattr(model, attribute)
    with torch.no_grad():
        with torch.autocast(
            device_type=model.device, dtype=model.dtype, enabled=model.use_amp
        ):
            for i in train_iterator:
                idx = batch_idx[i : (i + bs)]
                samples = array[idx]
                pred = model_function(samples)
                preds.append(pred)
            if i + bs < n_data:
                idx = batch_idx[(i + bs) :]
                samples = array[idx]
                pred = model_function(samples)
                preds.append(pred)
    preds = torch.cat(preds)
    return preds


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


# def plot(data, model, step_xy, step_t, name, trial, save=False):
#     try:
#         os.mkdir(name)
#     except:
#         pass

#     grid, n, p = setup_uniform_grid(data, step_xy)
#     vmin, vmax = -55, -46
#     extent = [grid[:, 0].min(), grid[:, 0].max(), grid[:, 1].min(), grid[:, 1].max()]
#     results = np.zeros_like(grid[:, 0])
#     time_nrm = model.data.nv_samples[0]
#     lat_nrm = model.data.nv_samples[1]
#     lon_nrm = model.data.nv_samples[2]
#     z_nrm = model.data.nv_targets[0]

#     grid[:, 0] = (grid[:, 0] - lat_nrm[0]) / lat_nrm[1]
#     grid[:, 1] = (grid[:, 1] - lon_nrm[0]) / lon_nrm[1]
#     xyt = np.column_stack([results.copy(), grid])

#     time = data[:, 0]
#     time_range = np.arange(time.min(), time.max(), step_t)
#     times = time_range.copy()
#     time_range = time_range.astype(int)
#     times = (times - time_nrm[0]) / time_nrm[1]
#     time_predictions = []
#     for it in range(times.shape[0]):
#         xyt[:, 0] = times[it]
#         tensor_xyt = torch.from_numpy(xyt).to(model.device, dtype=model.dtype)
#         prediction = predict(tensor_xyt, model)
#         prediction = prediction.to("cpu", dtype=torch.float64).numpy()
#         real_pred = prediction[:, 0] * z_nrm[1] + z_nrm[0]
#         results = real_pred
#         time_predictions.append(real_pred)
#         if save:
#             heatmap = results.copy().reshape(n, p, order="F")
#             heatmap[heatmap == 0] = np.nan
#             date = days_to_time_string(time_range[it])
#             plt.imshow(
#                 heatmap[::-1].T, extent=extent, vmin=vmin, vmax=vmax,
#             )
#             plt.xlabel("Lon")
#             plt.ylabel("Lat")
#             plt.title(f"Altitude at {date}")
#             plt.colorbar()
#             plt.savefig(f"{name}/heatmap_{date}.png")
#             plt.close()
#     time_predictions = np.column_stack(time_predictions)
#     results = time_predictions.std(axis=1) / step_t
#     time_std = results.copy().reshape(n, p, order="F")
#     time_std[time_std == 0] = np.nan
#     time_std = np.log(time_std + 1) / np.log(10)
#     plt.imshow(time_std[::-1].T, extent=extent)
#     plt.xlabel("Lon")
#     plt.ylabel("Lat")
#     plt.title("Pixel wise np.log(STD) per day")
#     plt.colorbar()
#     plt.savefig(f"{name}/time_std_trial_{trial}.png")
#     plt.close()
#     return time_predictions




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


def mae(target, prediction):
    return torch.absolute(target - prediction).mean().item()


def rmse(target, prediction):
    criterion = nn.MSELoss()
    target = target.to(dtype=float)
    prediction = prediction.to(dtype=float)
    loss = torch.sqrt(criterion(target, prediction)).item()
    return loss


def downsample_pointcloud(data, max_points=1000000):
    n_points = data.shape[0]
    
    if n_points <= max_points:
        return np.arange(n_points)
    
    indices = np.random.choice(n_points, max_points, replace=False)
    
    return indices

def plot_pc(y_gt, y_pred, input_data, nv_samples, suffix="", name="outputs"):

    try:
        os.mkdir(f"{name}/pc_{suffix}")
    except:
        pass
    input_data_n = input_data.copy()
    num = len(nv_samples)
    for i in range(num):
        input_data_n[:, i] = input_data[:, i] * nv_samples[i][1] + nv_samples[i][0]
    time_stamps = np.unique(input_data_n[:, 0])
    for t in time_stamps:   
        idx = np.where(input_data_n[:, 0] == t)[0]
        input_data_t = input_data_n[idx, :]
        y_pred_t = y_pred[idx]
        y_gt_t = y_gt[idx]


        vmin = -40 #min(min(y_pred_t), min(y_gt_t))
        vmax = -55 #max(y_pred_t.max(), y_gt_t.max())
        
        idx_2 = downsample_pointcloud(input_data_t, max_points=100000)
        input_data_t = input_data_t[idx_2, :]
        y_pred_t = y_pred_t[idx_2].squeeze()
        y_gt_t = y_gt_t[idx_2].squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(10, 30))
        scatter1 = axes[0].scatter(input_data_t[:, 1], input_data_t[:, 2], c=y_pred_t, s=0.01,
                                 cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Predicted Values')
        plt.colorbar(scatter1, ax=axes[0])
        
        scatter2 = axes[1].scatter(input_data_t[:, 1], input_data_t[:, 2], c=y_gt_t, s=0.01,
                                 cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth')
        plt.colorbar(scatter2, ax=axes[1])
        diff = y_gt_t - y_pred_t
        diff_log = np.log(np.abs(diff) + 1) * np.sign(diff)
        boundary = max(abs(diff_log.min()), abs(diff_log.max()))
        # Difference plot
        scatter3 = axes[2].scatter(input_data_t[:, 1], input_data_t[:, 2], c=diff_log, s=0.1,
                                vmin=-boundary, vmax=boundary, cmap='RdBu_r')
        axes[2].set_title('Difference (GT - pred)')
        plt.colorbar(scatter3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f"{name}/pc_{suffix}/comparison_time_{days_to_time_string(t)}_{suffix}.png")
        plt.close()



def evaluation(model, name, encoding):
    z_nrm = model.data.nv_targets[0]
    test_targets = model.test_set.targets * z_nrm[1] + z_nrm[0]
    test_predictions = predict(model.test_set.samples, model)
    test_predictions = test_predictions * z_nrm[1] + z_nrm[0]
    train_targets = model.data.targets * z_nrm[1] + z_nrm[0]
    train_predictions = predict(model.data.samples, model)
    train_predictions = train_predictions * z_nrm[1] + z_nrm[0]
    all_targets = torch.cat([train_targets, test_targets])
    all_predictions = torch.cat([train_predictions, test_predictions])
    # MAE computation
    mae_train = mae(train_targets, train_predictions)
    mae_test = mae(test_targets, test_predictions)
    mae_all = mae(all_targets, all_predictions)

    plot_pc(test_targets.flatten().cpu().float().numpy(), 
            test_predictions.flatten().cpu().float().numpy(), 
            model.test_set.samples.cpu().numpy(), model.data.nv_samples, suffix="validation", name=name)
    test_histo(test_targets.flatten().cpu().float().numpy(), 
                test_predictions.flatten().cpu().float().numpy(), 
                name=name, suffix="validation")
    plot_pc(train_targets.flatten().cpu().float().numpy(), 
            train_predictions.flatten().cpu().float().numpy(), 
            model.data.samples.cpu().numpy(), model.data.nv_samples, suffix="train", name=name)
    test_histo(train_targets.flatten().cpu().float().numpy(), 
            train_predictions.flatten().cpu().float().numpy(),
                name=name, suffix="train")
    # RMSE
    rmse_train = rmse(train_targets, train_predictions)
    rmse_test = rmse(test_targets, test_predictions)
    rmse_all = rmse(all_targets, all_predictions)

    return [mae_train, mae_test, mae_all, rmse_train, rmse_test, rmse_all]




def save_results(variables, name):
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
    results.to_csv(f"{name}/results.csv")


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

def days_to_time_string(days, reference_date="190101_000000"):
    # Parse the reference date
    ref_dt = datetime.strptime(reference_date, '%y%m%d_%H%M%S')
    
    # Calculate the target datetime by adding the days
    target_dt = ref_dt + timedelta(days=int(days))
    
    # Format the result as a string
    return target_dt.strftime('%y%m%d_%H%M%S')

def time_string_to_days(time_str, reference_date="190101_000000"):
    # Parse the input time string
    dt = datetime.strptime(time_str, '%y%m%d_%H%M%S')
    
    # Parse the reference date
    ref_dt = datetime.strptime(reference_date, '%y%m%d_%H%M%S')
    
    # Calculate the time difference in days
    time_difference = dt - ref_dt
    days = time_difference.total_seconds() / (24 * 3600)  # Convert seconds to days
    
    return days

def load_data(filename):
    with h5py.File(filename, "r") as f:
        time_str = 'dates_str'
        x_str = "x"
        y_str = "y"
        z_str = "z"
        time = list(f[time_str])
        time = [time_string_to_days(t) for t in time]
        x = list(f[x_str])
        y = list(f[y_str])
        z = list(f[z_str])
        data = np.column_stack((time, x, y, z))
        import pdb; pdb.set_trace()
    return data

def load_data_faster(opt = "default", path="/home/mletard/compute/4dinr/data"):
    if opt == 'default':
        filename = "data/data_simu.npy"
        return np.load(filename), None
    elif opt == "seasonal_beach_temporal":
        filename = path+"/seasonal_beach_temporal.npy"
        filename_index = path+"/seasonal_beach_temporal_split.npy"
    elif opt == "seasonal_beach_spatial":
        filename = path+"/seasonal_beach_spatial.npy"
        filename_index = path+"/seasonal_beach_spatial_split.npy"
    elif opt == "seasonal_beach_temporal_encoding":
        filename = path+"/seasonal_beach_temporal_encoding.npy"
        filename_index = path+"/seasonal_beach_temporal_encoding_split.npy"
    elif opt == "seasonal_beach_spatial_encoding":
        filename = path+"/seasonal_beach_spatial_encoding.npy"
        filename_index = path+"/seasonal_beach_spatial_encoding_split.npy"
    elif opt == "monthly_beach_temporal":
        filename = path+"/monthly_beach_temporal.npy"
        filename_index = path+"/monthly_beach_temporal_split.npy"
    elif opt == "monthly_beach_spatial":
        filename = path+"/monthly_beach_spatial.npy"
        filename_index = path+"/monthly_beach_spatial_split.npy"
    elif opt == "monthly_beach_temporal_encoding":
        filename = path+"/monthly_beach_temporal_encoding.npy"
        filename_index = path+"/monthly_beach_temporal_encoding_split.npy"
    elif opt == "monthly_beach_spatial_encoding":
        filename = path+"/monthly_beach_spatial_encoding.npy"
        filename_index = path+"/monthly_beach_spatial_encoding_split.npy"
    elif opt == "weekly_beach_temporal":
        filename = path+"/weekly_beach_temporal.npy"
        filename_index = path+"/weekly_beach_temporal_split.npy"
    elif opt == "weekly_beach_spatial":
        filename = path+"/weekly_beach_spatial.npy"
        filename_index = path+"/weekly_beach_spatial_split.npy"
    elif opt == "weekly_beach_temporal_encoding":
        filename = path+"/weekly_beach_temporal_encoding.npy"
        filename_index = path+"/weekly_beach_temporal_encoding_split.npy"
    elif opt == "weekly_beach_spatial_encoding":
        filename = path+"/weekly_beach_spatial_encoding.npy"
        filename_index = path+"/weekly_beach_spatial_encoding_split.npy"
    elif opt == "daily_beach_temporal":
        filename = path+"/daily_beach_temporal.npy"
        filename_index = path+"/daily_beach_temporal_split.npy"
    elif opt == "daily_beach_spatial":
        filename = path+"/daily_beach_spatial.npy"
        filename_index = path+"/daily_beach_spatial_split.npy"
    elif opt == "daily_beach_temporal_encoding":
        filename = path+"/daily_beach_temporal_encoding.npy"
        filename_index = path+"/daily_beach_temporal_encoding_split.npy"
    elif opt == "daily_beach_spatial_encoding":
        filename = path+"/daily_beach_spatial_encoding.npy"
        filename_index = path+"/daily_beach_spatial_encoding_split.npy"
    elif opt == "final_map":
        filename = path+"map.npy"
        filename_index = path+"map_split.npy"
    elif opt == "time_series":
        filename = path+"timeseries.npy"
        filename_index = path+"timeseries_split.npy"

    return np.load(filename), np.load(filename_index)

def split_test(data, index):
    if index is None:
        return data, data[:1000].copy(), None
    idx = (index == 2).squeeze()
    data_test = data[idx, :]
    data_train = data[~idx, :]
    return data_train, data_test, index[~idx]

def evaluation_test(model, data, name, encoding):
    if not encoding:
        data_txy = data[:, :3].copy()
        test_targets = data[:, 3:4]
    else:
        data_txy = data[:, :12].copy()
        test_targets = data[:, 12:13]
    model.test_set.normalize(data_txy, model.test_set.nv_samples, True)
    z_pred = predict(torch.tensor(data_txy).cuda().float(), model)
    z_nrm = model.data.nv_targets[0]
    test_pred = z_pred * z_nrm[1] + z_nrm[0]
    mae_test = mae(torch.tensor(test_targets).cuda(), test_pred)
    plot_pc(test_targets, 
                test_pred.flatten().cpu().float().numpy(), 
                data_txy, model.data.nv_samples, suffix="test", name=name)
    test_histo(test_targets, 
                test_pred.flatten().cpu().float().numpy(), name, suffix="test")
    # RMSE
    rmse_test = rmse(torch.tensor(test_targets).cuda(), test_pred)

    return [mae_test, rmse_test]

def test_histo(ztrue, zpred, name, suffix=''):

    errors = ztrue.squeeze() - zpred.squeeze()
    lower_percentile = 1
    upper_percentile = 99
    low, high = np.percentile(errors, [lower_percentile, upper_percentile])
    filtered_errors = errors[(errors >= low) & (errors <= high)]
    cmap = plt.cm.RdBu_r
    norm_full = plt.Normalize(-max(abs(errors.min()), abs(errors.max())), 
                            max(abs(errors.min()), abs(errors.max())))
    
    norm_filtered = plt.Normalize(-max(abs(filtered_errors.min()), abs(filtered_errors.max())), 
                                max(abs(filtered_errors.min()), abs(filtered_errors.max())))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme complet
    counts, bins, patches = axes[0].hist(errors, bins=30, edgecolor="black", alpha=0.9)
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        center = 0.5 * (left + right)
        try:
            for bar in patch:
                bar.set_facecolor(cmap(norm_full(center)))
        except:
            patch.set_facecolor(cmap(norm_full(center)))
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Histogram of Prediction Errors (full)", fontsize=12, weight="bold")
    axes[0].set_xlabel("Error (y_true - y_pred)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # Histogramme filtré par percentiles
    counts_f, bins_f, patches_f = axes[1].hist(filtered_errors, bins=30, edgecolor="black", alpha=0.9)
    for patch, left, right in zip(patches_f, bins_f[:-1], bins_f[1:]):
        center = 0.5 * (left + right)
        # Do:
        # for bar in patch:
        #     bar.set_facecolor(cmap(norm_filtered(center)))
        patch.set_facecolor(cmap(norm_filtered(center)))
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"Histogram of Prediction Errors ({lower_percentile}-{upper_percentile} percentile)", 
                    fontsize=12, weight="bold")
    axes[1].set_xlabel("Error (y_true - y_pred)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name}/pc_{suffix}/histo_errors_time_{suffix}.png")
    plt.close()


def sample_hp(hp, trial):
    hp.lr = trial.suggest_float(
                    "lr",
                    1e-4,
                    1e-2,
                    log=True,
                )
    hp.model["scale"] = trial.suggest_float("scale", 1e-2, 5e-1, log=True)
  
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

    model_hp.input_size = int(npz["input_size"])
    model_hp.output_size = int(npz["output_size"])
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
    scores_id = get_n_best_trials(study)
    id_trial = scores_id[-1][0]
    npz = f"{opt.name}/multiple/optuna_{id_trial}.npz"
    weights = f"{opt.name}/multiple/optuna_{id_trial}.pth"
    model_hp.device = "cuda" if model_hp.gpu else "cpu"
    NN = load_model(model_hp, weights, npz, data, index, encoding)
    # time_preds = plot(data, NN, opt.name, 0, True)  # 0 is trial
    metrics = evaluation(NN, opt.name, encoding)
    metrics_test = evaluation_test(NN, data_test, opt.name, encoding)
    # import pdb; pdb.set_trace()
    save_results(metrics + metrics_test, opt.name)
    # plot_NN(NN, model_hp, opt.name)


if __name__ == "__main__":
    main()
