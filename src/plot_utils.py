import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta


def downsample_pointcloud(data, max_points=1000000):
    n_points = data.shape[0]
    if n_points <= max_points:
        return np.arange(n_points)
    indices = np.random.choice(n_points, max_points, replace=False)
    return indices


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

        low_pred, high_pred = np.percentile(y_pred_t, [5, 95])
        low_true, high_true = np.percentile(y_gt_t, [5, 95])
        vmin = min(low_pred, low_true)
        vmax = max(high_pred, high_true)
        
        idx_2 = downsample_pointcloud(input_data_t, max_points=100000)
        input_data_t = input_data_t[idx_2, :]
        y_pred_t = y_pred_t[idx_2].squeeze()
        y_gt_t = y_gt_t[idx_2].squeeze()

        fig, axes = plt.subplots(1, 4, figsize=(10, 30))
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
        low, high = np.percentile(diff, [5, 95])
        boundary = max(abs(low), abs(high))
        # Difference plot
        scatter3 = axes[2].scatter(input_data_t[:, 1], input_data_t[:, 2], c=diff, s=0.1,
                                vmin=-boundary, vmax=boundary, cmap='RdBu_r')
        axes[2].set_title('Difference (GT - pred)')
        plt.colorbar(scatter3, ax=axes[2])

        boundary = max(abs(diff_log.min()), abs(diff_log.max()))
        scatter4 = axes[3].scatter(input_data_t[:, 1], input_data_t[:, 2], c=diff_log, s=0.1,
                                vmin=-boundary, vmax=boundary, cmap='RdBu_r')
        axes[3].set_title('Log Difference (GT - pred)')
        plt.colorbar(scatter4, ax=axes[3])
        
        plt.tight_layout()
        plt.savefig(f"{name}/pc_{suffix}/comparison_time_{days_to_time_string(t)}_{suffix}.png")
        plt.close()


def plot_feature(rough_gt, rough_pred, input_data, nv_samples, scale, percentile=5, suffix="", name="outputs", feat_name="", down=True):
    input_data_n = input_data.copy()
    num = len(nv_samples)
    for i in range(num):
        input_data_n[:, i] = input_data[:, i] * nv_samples[i][1] + nv_samples[i][0]
    time_stamps = np.unique(input_data_n[:, 0])
    for t in time_stamps:   
        idx = np.where(input_data_n[:, 0] == t)[0]
        input_data_t = input_data_n[idx, :]
        y_pred_t = rough_pred[idx]
        y_gt_t = rough_gt[idx]
        
        if down:
            idx_2 = downsample_pointcloud(input_data_t, max_points=100000)
            input_data_t = input_data_t[idx_2, :]
            y_pred_t = y_pred_t[idx_2].squeeze()
            y_gt_t = y_gt_t[idx_2].squeeze()
        else:
            y_pred_t = y_pred_t[:].squeeze()
            y_gt_t = y_gt_t[:].squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(10, 30))
        low, high = np.percentile(y_pred_t[~np.isnan(y_pred_t)], [percentile, 100-percentile])
        scatter1 = axes[0].scatter(input_data_t[~np.isnan(y_pred_t), 1], input_data_t[~np.isnan(y_pred_t), 2], c=y_pred_t[~np.isnan(y_pred_t)], s=0.01, vmin=low, vmax=high,
                                 cmap='viridis')
        axes[0].set_title(f"{feat_name} of predicted surface")
        plt.colorbar(scatter1, ax=axes[0])
        
        low, high = np.percentile(y_gt_t[~np.isnan(y_gt_t)], [percentile, 100-percentile])
        scatter2 = axes[1].scatter(input_data_t[~np.isnan(y_gt_t), 1], input_data_t[~np.isnan(y_gt_t), 2], c=y_gt_t[~np.isnan(y_gt_t)], s=0.01, vmin=low, vmax=high,
                                 cmap='viridis')
        axes[1].set_title(f"{feat_name} of Ground Truth point cloud")
        plt.colorbar(scatter2, ax=axes[1])
        diff = y_gt_t - y_pred_t
        # diff_log = np.log(np.abs(diff) + 1) * np.sign(diff)
        low, high = np.percentile(diff[~np.isnan(diff)], [percentile, 100-percentile])
        # boundary = max(abs(diff.min()), abs(diff.max()))
        boundary = max(abs(low), abs(high))
        # Difference plot
        scatter3 = axes[2].scatter(input_data_t[~np.isnan(diff), 1], input_data_t[~np.isnan(diff), 2], c=diff[~np.isnan(diff)], s=0.1,
                                vmin=-boundary, vmax=boundary, cmap='RdBu_r')
        axes[2].set_title(f"Difference of {feat_name} (GT - pred)")
        plt.colorbar(scatter3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f"{name}/pc_{suffix}/{feat_name}_time_{days_to_time_string(t)}_{suffix}_{str(scale)}.png")
        plt.close()


def plot_error_change(error, change_value, t, name, suffix="", percentile=1):
    try:
        os.mkdir(f"{name}/pc_{suffix}/change_errors")
    except:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _, xmax = np.percentile(error[~np.isnan(change_value[:,0])], [percentile, 100-percentile])
    _, ymax = np.percentile(change_value[~np.isnan(change_value[:,0]),0], [percentile, 100-percentile])
    lims = [
        min(min(error[~np.isnan(change_value[:,0])]), min(change_value[~np.isnan(change_value[:,0]),0])),
        max(xmax, ymax),
    ]
    axes[0].scatter(error[~np.isnan(change_value[:,0])], change_value[~np.isnan(change_value[:,0]),0], s=1, color='green')
    axes[0].plot(lims, lims, 'k--', label='x=y')
    axes[0].set_title("Absolute error depending on absolute change before considered date", fontsize=12, weight="bold")
    axes[0].set_xlabel("|error| (m)")
    axes[0].set_ylabel("|change| (m)")
    axes[0].set_aspect('equal')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    
    _, xmax = np.percentile(error[~np.isnan(change_value[:,1])], [percentile, 100-percentile])
    _, ymax = np.percentile(change_value[~np.isnan(change_value[:,1]),1], [percentile, 100-percentile])
    lims = [
        min(min(error[~np.isnan(change_value[:,1])]), min(change_value[~np.isnan(change_value[:,1]),1])),
        max(xmax, ymax),
    ]
    axes[1].scatter(error[~np.isnan(change_value[:,1])], change_value[~np.isnan(change_value[:,1]),1], s=1, color='green')
    axes[1].plot(lims, lims, 'k--', label='x=y')
    axes[1].set_title("Absolute error depending on absolute change after considered date", fontsize=12, weight="bold")
    axes[1].set_xlabel("|error| (m)")
    axes[1].set_ylabel("|change| (m)")
    axes[1].set_aspect('equal')
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    plt.tight_layout()
    plt.savefig(f"{name}/pc_{suffix}/change_errors/change_errors_time_{days_to_time_string(t)}_{suffix}.png")
    plt.close()


def plot_timeseries(elevations_true, ts_pred, tdays_pred, timestamps_true, corepoints, coords, cp_idx_sel, name, folder, suffix=""):
    try:
        os.mkdir(f"{name}/pc_{suffix}/{folder}")
    except:
        pass
    
    plt.rcParams.update({
        'font.size': 22,          # base font size
        'axes.titlesize': 20,     # title font size
        'axes.labelsize': 18,     # x/y labels
        'lines.linewidth': 3,     # line width
        'lines.markersize': 2,   # default marker size
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16
    })
    fig = plt.figure(figsize=(20,7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])#, projection='3d', computed_zorder=False)
    ax2 = fig.add_subplot(gs[1])

    distances_epoch = np.array(([d[0] for d in elevations_true]))
    timeseries_sel = elevations_true[cp_idx_sel]
    idx_2 = downsample_pointcloud(corepoints, max_points=100000)
    corepoints = corepoints[idx_2]
    distances_epoch = distances_epoch[idx_2]
    d = ax1.scatter(corepoints[:,1], corepoints[:,0], c=distances_epoch[:], cmap='viridis', s=1, zorder=1)
    plt.colorbar(d, format=('%.2f'), label='Elevation [m]', ax=ax1, shrink=.5, pad=.15, orientation='horizontal')
    ax1.scatter(coords[1], coords[0], c='white', s=300, zorder=2, label='Selected location', marker='*',facecolors='white', edgecolors='black', linewidths=1)
    ax1.legend()
    ax1.tick_params(labelbottom=False, labelleft=False)
    ax1.set_xlabel('Y [m]')
    ax1.set_ylabel('X [m]')
    ax1.set_aspect('equal')
    ax1.set_title('Elevation at %s' % (timestamps_true[0]))

    ax2.plot(timestamps_true[~np.isnan(timeseries_sel)], timeseries_sel[~np.isnan(timeseries_sel)], color='lightgray', alpha=0.3, linestyle='-', zorder=1)
    pred_order = np.argsort(tdays_pred)
    tdays_pred = np.array((tdays_pred))
    ax2.plot(tdays_pred[pred_order], ts_pred[pred_order], color='gray', linestyle='-', zorder=1)
    ax2.scatter(timestamps_true[~np.isnan(timeseries_sel)], timeseries_sel[~np.isnan(timeseries_sel)], c=timeseries_sel[~np.isnan(timeseries_sel)], s=150, cmap='viridis', zorder=2, label='Measured')
    ax2.scatter(tdays_pred[pred_order], ts_pred[pred_order], c=ts_pred[pred_order], marker="x", s=150, cmap='viridis', zorder=2, label='Predicted')
    ax2.legend()
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_title('Time series at selected location')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{name}/pc_{suffix}/{folder}/time_series_plot_{suffix}_{str(cp_idx_sel)}.png")


def plot_timeseries_uncert(elevations_true, ts_pred, zmean, zstd, tdays_pred, timestamps_true, tmean, cp_idx_sel, name, folder, suffix=""):
    try:
        os.mkdir(f"{name}/pc_{suffix}/{folder}")
    except:
        pass
    
    plt.rcParams.update({
        'font.size': 22,          # base font size
        'axes.titlesize': 20,     # title font size
        'axes.labelsize': 18,     # x/y labels
        'lines.linewidth': 3,     # line width
        'lines.markersize': 2,   # default marker size
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16
    })
    # axes = plt.figure(figsize=(20,7))
    fig, axes = plt.subplots(figsize=(10, 5))
    zmean = zmean[...,2:]
    zstd = zstd[..., 2:]
    order = np.argsort(tmean)
    plt.fill_between(
        pd.to_datetime(tmean[order]),
        zmean[cp_idx_sel, order] - zstd[cp_idx_sel, order],
        zmean[cp_idx_sel, order] + zstd[cp_idx_sel, order],
        color="lightgray", #'skyblue',
        alpha=0.3,
        label='±1σ'
    )

    timeseries_sel = elevations_true[cp_idx_sel]

    # ax2.plot(timestamps_true[~np.isnan(timeseries_sel)], timeseries_sel[~np.isnan(timeseries_sel)], color='gray', linestyle='-', zorder=1)
    pred_order = np.argsort(tdays_pred)
    tdays_pred = np.array((tdays_pred))
    axes.plot(tdays_pred[pred_order], ts_pred[pred_order], color='gray', linestyle='-', zorder=1)
    axes.scatter(timestamps_true[~np.isnan(timeseries_sel)], timeseries_sel[~np.isnan(timeseries_sel)], c=timeseries_sel[~np.isnan(timeseries_sel)], s=150, cmap='viridis', zorder=2, label='Measured')
    axes.scatter(tdays_pred[pred_order], ts_pred[pred_order], c=ts_pred[pred_order], marker="x", s=150, cmap='viridis', zorder=2, label='Predicted')
    axes.legend()
    axes.set_xlabel('Date')
    axes.set_ylabel('Elevation [m]')
    axes.set_title('Time series at selected location')
    axes.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{name}/pc_{suffix}/{folder}/time_series_plot_{suffix}_{str(cp_idx_sel)}.png")


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