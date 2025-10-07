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
import py4dgeo
from pc_utils import *
from plot_utils import *
from temp_encoding import *


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

def load_eval_data_faster(opt, path="/home/mletard/compute/4dinr/data"):
    if "seasonal_beach" in opt:
        change_data = path+"/bitemporal_change_seasonal_beach.npy"
        time_series = path+"/seasonal_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/seasonal_beach.zip")
    elif "monthly_beach" in opt:
        change_data = path+"/bitemporal_change_monthly_beach.npy"
        time_series = path+"/monthly_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/monthly_beach.zip")
    elif "weekly_beach" in opt:
        change_data = path+"/bitemporal_change_weekly_beach.npy"
        time_series = path+"/weekly_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/weekly_beach.zip")
    elif "daily_beach" in opt:
        change_data = path+"/bitemporal_change_daily_beach.npy"
        time_series = path+"/daily_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/daily_beach.zip")

    return np.load(change_data), np.load(time_series), time_series_gt


def split_test(data, index):
    if index is None:
        return data, data[:1000].copy(), None
    idx = (index == 2).squeeze()
    data_test = data[idx, :]
    data_train = data[~idx, :]
    return data_train, data_test, index[~idx]

def evaluation_test(model, data, name, encoding):
    if not encoding:
        data_txy = data[:, [0,-3,-2]].copy()
        test_targets = data[:, -1:]
    else:
        data_txy = data[:, 1:-1].copy()
        test_targets = data[:, -1:]
    
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

    test_pred = test_pred.flatten().cpu().float().numpy()
    pred_xyz = np.concatenate((data[:,[-3,-2]], test_pred.reshape((-1,1))), axis=1)
    scale = 1.0
    rough_gt, dzdx_true, dzdy_true = get_roughness(data[:, [-3,-2,-1]], data[:, [-3,-2,-1]], scale)
    rough_pred, dzdx_pred, dzdy_pred = get_roughness(pred_xyz[:], pred_xyz, scale) #::10
    plot_feature(rough_gt, rough_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="Roughness")
    plot_feature(dzdx_true, dzdx_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="DzDx")
    plot_feature(dzdy_true, dzdy_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="DzDy")
    scale = 3.0
    rough_gt, dzdx_true, dzdy_true = get_roughness(data[:, [-3,-2,-1]], data[:, [-3,-2,-1]], scale)
    rough_pred, dzdx_pred, dzdy_pred = get_roughness(pred_xyz[:], pred_xyz, scale) #::10
    plot_feature(rough_gt, rough_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="Roughness")
    plot_feature(dzdx_true, dzdx_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="DzDx")
    plot_feature(dzdy_true, dzdy_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="DzDy")
    scale = 5.0
    rough_gt, dzdx_true, dzdy_true = get_roughness(data[:, [-3,-2,-1]], data[:, [-3,-2,-1]], scale)
    rough_pred, dzdx_pred, dzdy_pred = get_roughness(pred_xyz[:], pred_xyz, scale) #::10
    plot_feature(rough_gt, rough_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="Roughness")
    plot_feature(dzdx_true, dzdx_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="DzDx")
    plot_feature(dzdy_true, dzdy_pred, data_txy, model.data.nv_samples, scale, suffix="test", name=name, feat_name="DzDy")
    return [mae_test, rmse_test]


def evaluation_with_change(model, change_data, name, encoding):
    data = change_data[:,:-2]
    change = change_data[:,-2:]
    if not encoding:
        data_txy = data[:, [0,-3,-2]].copy()
        test_targets = data[:, -1:]
    else:
        data_txy = data[:, 1:-1].copy()
        test_targets = data[:, -1:]
    model.test_set.normalize(data_txy, model.test_set.nv_samples, True)
    z_pred = predict(torch.tensor(data_txy).cuda().float(), model)
    z_nrm = model.data.nv_targets[0]
    test_pred = z_pred * z_nrm[1] + z_nrm[0]
    error_test = torch.absolute(torch.tensor(test_targets).cuda() - test_pred)
    # error_test = torch.tensor(test_targets).cuda() - test_pred

    dates = np.unique(data_txy[:,0])
    for t in dates:
        t_ = t * model.test_set.nv_samples[0][1] + model.test_set.nv_samples[0][0]
        test_date = np.where(data_txy[:,0]== t)[0]
        plot_error_change(error_test[test_date].cpu().numpy().flatten(), np.abs(change[test_date]), t_, name, "test")
    plot_error_change(error_test.cpu().numpy().flatten(), np.abs(change), 0, name, "test", percentile=0)
    #plot point-wise error with respect to change that occurred at that point


def evaluation_timeseries(model, data, ts_gt, name, encoding, n_plots=10, suffix="test"):
    elevations_true = ts_gt.distances
    timestamps = np.array(([t + ts_gt.reference_epoch.timestamp for t in ts_gt.timedeltas]))
    corepoints = ts_gt.corepoints.cloud
    ref_dt = datetime.strptime("190101_000000", '%y%m%d_%H%M%S')

    full_valid_id = np.where(np.any(np.isnan(elevations_true), axis=1) == False)[0]
    nb_nans = np.count_nonzero(np.isnan(elevations_true), axis=1)
    gap_filling_id = np.where((nb_nans >= 0.4 * elevations_true.shape[1])&(nb_nans <= 0.6 * elevations_true.shape[1]))[0]

    half_res = np.array((timestamps[1:]-timestamps[:-1])) /2
    newdates = []
    for srd in range(timestamps.shape[0]-1):
        newdates.append(timestamps[srd] + half_res[srd])

    encod_newdates, days_el_newdates = encode_time_info(Time(newdates))
    super_res_id_sel = np.random.choice(full_valid_id, n_plots, replace=False)
    super_res_txyz = np.empty((0,14))
    for sr_id in super_res_id_sel:
        coords = ts_gt.corepoints.cloud[sr_id,[0,1,2]].reshape((1,-1))
        new_xyz = np.repeat(coords, encod_newdates.shape[0],axis=0)
        new_xyz[:,-1] = np.NaN
        new_txyz = np.concatenate((days_el_newdates.reshape((-1,1)), encod_newdates, new_xyz), axis=1)
        super_res_txyz = np.append(super_res_txyz, new_txyz, axis=0)
    
    if not encoding:
        data_txy = np.concatenate((data[:, [0,-3,-2]].copy(), super_res_txyz[:,[0,-3,-2]].copy()),axis=0)
        raw_txy = data_txy.copy()
        print(data.shape, data_txy.shape)
    else:
        data_txy = np.concatenate((data[:, 1:-1].copy(), super_res_txyz[:,1:-1].copy()), axis=0)
        raw_txy = data_txy.copy()
    
    model.test_set.normalize(data_txy, model.test_set.nv_samples, True)
    z_pred = predict(torch.tensor(data_txy).cuda().float(), model)
    z_nrm = model.data.nv_targets[0]
    test_pred = z_pred * z_nrm[1] + z_nrm[0]
    ts_pred = test_pred.cpu()

    full_valid_id_sel = np.random.choice(full_valid_id, n_plots, replace=False)
    for v_id in full_valid_id_sel:
        coords = ts_gt.corepoints.cloud[v_id,[0,1]]
        pt_id = np.where((data[:,-3]==coords[0])&(data[:,-2]==coords[1]))[0]
        pred_ = ts_pred[pt_id]
        target_dt=[]
        for i in pt_id:
            target_dt.append(ref_dt + timedelta(days=int(data[i,0])))
        plot_timeseries(elevations_true, pred_, target_dt, timestamps, corepoints, coords, v_id, name, 'time_series_eval', "test")

    # super_res_id_sel = np.random.choice(full_valid_id, n_plots, replace=False)
    for sr_id in super_res_id_sel:
        coords = ts_gt.corepoints.cloud[sr_id,[0,1]]
        pt_id = np.where((raw_txy[:,-2]==coords[0])&(raw_txy[:,-1]==coords[1]))[0]
        pred_ = ts_pred[pt_id]
        target_dt=[]
        for i in pt_id:
            target_dt.append(ref_dt + timedelta(days=int(raw_txy[i,0])))
        plot_timeseries(elevations_true, pred_, target_dt, timestamps, corepoints, coords, sr_id, name, 'temp_super_res', "test")
    
    gap_fill_id_sel = np.random.choice(gap_filling_id, n_plots, replace=False)
    for gf_id in gap_fill_id_sel:
        coords = ts_gt.corepoints.cloud[gf_id,[0,1]]
        pt_id = np.where((data[:,-3]==coords[0])&(data[:,-2]==coords[1]))[0]
        pred_ = ts_pred[pt_id]
        target_dt=[]
        for i in pt_id:
            target_dt.append(ref_dt + timedelta(days=int(data[i,0])))
        plot_timeseries(elevations_true, pred_, target_dt, timestamps, corepoints, coords, gf_id, name, 'temp_gap_filling', "test")
    
    #plot pred time series + true time series
    #compute time series metrics


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
    print("\nBest trial is trial ", id_trial)
    npz = f"{opt.name}/multiple/optuna_{id_trial}.npz"
    weights = f"{opt.name}/multiple/optuna_{id_trial}.pth"
    model_hp.device = "cuda" if model_hp.gpu else "cpu"
    NN = load_model(model_hp, weights, npz, data, index, encoding)
    # time_preds = plot(data, NN, opt.name, 0, True)  # 0 is trial
    metrics = evaluation(NN, opt.name, encoding)
    metrics_test = evaluation_test(NN, data_test, opt.name, encoding)
    change_data, ts_pts, ts_gt = load_eval_data_faster(keyword)
    evaluation_with_change(NN, change_data, opt.name, encoding)
    evaluation_timeseries(NN, ts_pts, ts_gt, opt.name, encoding)
    # import pdb; pdb.set_trace()
    save_results(metrics + metrics_test, opt.name)
    plot_NN(NN, model_hp, opt.name)


if __name__ == "__main__":
    main()
