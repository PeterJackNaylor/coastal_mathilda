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
from dataloader import return_dataset, load_data_faster, load_eval_data_faster
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


def evaluation_test(model, data, name, encoding, feats=True, suffix="test"):
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
                data_txy, model.data.nv_samples, suffix=suffix, name=name)
    test_histo(test_targets, 
                test_pred.flatten().cpu().float().numpy(), name, suffix=suffix)
    # RMSE
    rmse_test = rmse(torch.tensor(test_targets).cuda(), test_pred)

    if feats:
        test_pred = test_pred.flatten().cpu().float().numpy()
        true_xyz = data[:, [-3,-2,-1]]
        pred_xyz = np.concatenate((data[:,[-3,-2]], test_pred.reshape((-1,1))), axis=1)
        scale = 1.0
        idx_2 = downsample_pointcloud(true_xyz, max_points=100000)
        idx_2pred = downsample_pointcloud(pred_xyz, max_points=100000)
        corepoints = true_xyz[idx_2]
        corepoints_pred = pred_xyz[idx_2pred]
        rough_gt, dzdx_true, dzdy_true = get_roughness(corepoints, true_xyz, scale)
        rough_pred, dzdx_pred, dzdy_pred = get_roughness(corepoints_pred, pred_xyz, scale) #::10
        plot_feature(rough_gt, rough_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="Roughness", down=False)
        plot_feature(dzdx_true, dzdx_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="DzDx", down=False)
        plot_feature(dzdy_true, dzdy_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="DzDy", down=False)
        scale = 3.0
        rough_gt, dzdx_true, dzdy_true = get_roughness(corepoints, true_xyz, scale)
        rough_pred, dzdx_pred, dzdy_pred = get_roughness(corepoints_pred, pred_xyz, scale) #::10
        plot_feature(rough_gt, rough_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="Roughness", down=False)
        plot_feature(dzdx_true, dzdx_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="DzDx", down=False)
        plot_feature(dzdy_true, dzdy_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="DzDy", down=False)
        scale = 5.0
        rough_gt, dzdx_true, dzdy_true = get_roughness(corepoints, true_xyz, scale)
        rough_pred, dzdx_pred, dzdy_pred = get_roughness(corepoints_pred, pred_xyz, scale) #::10
        plot_feature(rough_gt, rough_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="Roughness", down=False)
        plot_feature(dzdx_true, dzdx_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="DzDx", down=False)
        plot_feature(dzdy_true, dzdy_pred, data_txy[idx_2], model.data.nv_samples, scale, suffix=suffix, name=name, feat_name="DzDy", down=False)
    return [mae_test, rmse_test]


def evaluation_with_change(model, change_data, name, encoding, suffix="test"):
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
        plot_error_change(error_test[test_date].cpu().numpy().flatten(), np.abs(change[test_date]), t_, name, suffix)
    plot_error_change(error_test.cpu().numpy().flatten(), np.abs(change), 0, name, suffix, percentile=0)
    #plot point-wise error with respect to change that occurred at that point


# def evaluation_timeseries(model, data, ts_gt, uncert_data, uncert_time, zmean, zstd, tmean, name, encoding, n_plots=10, suffix="test"):
def evaluation_timeseries(model, data, ts_gt, uncert_data, uncert_time, name, encoding, n_plots=10, suffix="test"):
    elevations_true = ts_gt.distances
    elevations_uncertainty = uncert_data
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
        plot_timeseries(elevations_true, pred_, target_dt, timestamps, corepoints, coords, v_id, name, 'time_series_eval', suffix)
        # plot_timeseries(elevations_uncertainty[:,2:], pred_, target_dt, uncert_time, corepoints, coords, v_id, name, 'time_series_uncert', "test")
        # plot_timeseries_uncert(elevations_true, pred_, zmean, zstd, target_dt, timestamps, tmean, v_id, name, 'time_series_uncert2', "test")

    # super_res_id_sel = np.random.choice(full_valid_id, n_plots, replace=False)
    for sr_id in super_res_id_sel:
        coords = ts_gt.corepoints.cloud[sr_id,[0,1]]
        pt_id = np.where((raw_txy[:,-2]==coords[0])&(raw_txy[:,-1]==coords[1]))[0]
        pred_ = ts_pred[pt_id]
        target_dt=[]
        for i in pt_id:
            target_dt.append(ref_dt + timedelta(days=int(raw_txy[i,0])))
        plot_timeseries(elevations_true, pred_, target_dt, timestamps, corepoints, coords, sr_id, name, 'temp_super_res', suffix)
    
    gap_fill_id_sel = np.random.choice(gap_filling_id, n_plots, replace=False)
    for gf_id in gap_fill_id_sel:
        coords = ts_gt.corepoints.cloud[gf_id,[0,1]]
        pt_id = np.where((data[:,-3]==coords[0])&(data[:,-2]==coords[1]))[0]
        pred_ = ts_pred[pt_id]
        target_dt=[]
        for i in pt_id:
            target_dt.append(ref_dt + timedelta(days=int(data[i,0])))
        plot_timeseries(elevations_true, pred_, target_dt, timestamps, corepoints, coords, gf_id, name, 'temp_gap_filling', suffix)
    
    #plot pred time series + true time series
    #compute time series metrics