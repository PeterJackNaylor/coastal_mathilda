import numpy as np
import py4dgeo
import os
from datetime import datetime
from astropy.time import Time
import py4dgeo
import matplotlib.pyplot as plt
from temp_encoding import *


def gt_elevation_ts(data_path, save_path, corepoints_path, name, make_analysis = False):
    analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/{name}.zip')
    if make_analysis:
        core_pc = py4dgeo.read_from_las(corepoints_path)
        corepoints = core_pc.cloud
        corept_dummy = np.zeros(corepoints.shape)
        corept_dummy[:,:-1] = corepoints[:,:2]

        pc_list = []
        files = os.listdir(data_path)
        for ff in files:
            if ".laz" in ff or ".las" in ff:
                pc_list.append(os.path.join(data_path, ff))
        print(len(pc_list), " point clouds.")
        timestamps = []
        for f in pc_list:
            timestamp_str = '_'.join(f.split('.')[0].split('/')[-1:]) # yields YYMMDD_hhmmss
            timestamp = datetime.strptime(timestamp_str, '%y%m%d_%H%M%S')
            timestamps.append(timestamp)
        asort = np.argsort(timestamps)
        analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/{name}.zip', force=True)
        epoch_dummy = py4dgeo.Epoch(cloud=corept_dummy, normals=None, timestamp=datetime.strptime('190901_000000', '%y%m%d_%H%M%S'))
        reference_epoch = epoch_dummy
        reference_epoch.timestamp = datetime.strptime('190901_000000', '%y%m%d_%H%M%S')
        analysis.reference_epoch = reference_epoch
        analysis.corepoints = corepoints
        analysis.m3c2 = py4dgeo.M3C2(cyl_radius=0.25, corepoint_normals=np.tile([0, 0, 1], (corepoints.shape[0], 1)), max_distance=60.0, registration_error = 0.0)
        epochs = []
        for k in range(0,asort.shape[0],1):
            pc_id = asort[k]
            epoch = py4dgeo.read_from_las(pc_list[pc_id])
            epoch.timestamp = timestamps[pc_id]
            epochs.append(epoch)
        analysis.add_epochs(*epochs)
        print(np.all(np.isnan(analysis.distances)))

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
    corepoints = analysis.corepoints.cloud
    distances = analysis.distances
    distances_epoch = [d[1] for d in distances]
    valid_id = np.where(np.any(np.isnan(distances), axis=1) == False)[0]
    cp_idx_sel = np.random.choice(valid_id, 1)[0]
    print(cp_idx_sel)
    coord_sel = analysis.corepoints.cloud[cp_idx_sel]
    timeseries_sel = distances[cp_idx_sel]
    # print(np.where(np.any(np.isnan(distances), axis=1) == False)[0])
    print(timeseries_sel)
    timestamps = [t + analysis.reference_epoch.timestamp for t in analysis.timedeltas]
    # d = ax1.scatter(corepoints[:,0], corepoints[:,1], corepoints[:,2], c=distances_epoch[:], cmap='viridis', s=1, zorder=1)
    d = ax1.scatter(corepoints[:,1], corepoints[:,0], c=distances_epoch[:], cmap='viridis', s=1, zorder=1)
    plt.colorbar(d, format=('%.2f'), label='Elevation [m]', ax=ax1, shrink=.5, pad=.15, orientation='horizontal')
    # ax1.scatter(coord_sel[0], coord_sel[1], coord_sel[2], c='white', s=50, zorder=2, label='Selected location', marker='*',facecolors='white', edgecolors='black', linewidths=0.5)
    ax1.scatter(coord_sel[1], coord_sel[0], c='white', s=300, zorder=2, label='Selected location', marker='*',facecolors='white', edgecolors='black', linewidths=1)
    ax1.legend()
    ax1.tick_params(labelbottom=False, labelleft=False)
    ax1.set_xlabel('Y [m]')
    ax1.set_ylabel('X [m]')
    # ax1.set_zlabel('Z [m]')
    ax1.set_aspect('equal')
    # ax1.view_init(elev=30., azim=165.)
    ax1.set_title('Elevation at %s' % (analysis.reference_epoch.timestamp+analysis.timedeltas[1]))
    ax2.plot(timestamps, timeseries_sel, color='gray', linestyle='-', zorder=1)
    ax2.scatter(timestamps, timeseries_sel, c=timeseries_sel, s=150, cmap='viridis', zorder=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_title('Time series at selected location')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/time_series_plot_{name}_{str(cp_idx_sel)}.png')
    # plt.show()


def pair_m3c2(data_path, save_path, name):
    pc_list = []
    files = os.listdir(data_path)
    for ff in files:
        if ".laz" in ff or ".las" in ff:
            pc_list.append(os.path.join(data_path, ff))
    print(len(pc_list), " point clouds.")
    timestamps = []
    for f in pc_list:
        timestamp_str = '_'.join(f.split('.')[0].split('/')[-1:]) # yields YYMMDD_hhmmss
        timestamp = datetime.strptime(timestamp_str, '%y%m%d_%H%M%S')
        timestamps.append(timestamp)

    encodings, days_el = encode_time_info(Time(timestamps)) #not sorted
    asort = np.argsort(timestamps)
    distances = np.empty((0, 2))
    coordinates = np.empty((0,3))
    t = np.empty((0,1))
    encoding = np.empty((0,10))
    for k in range(1,asort.shape[0]-1,1):
        ref_id = asort[k]
        before_id = asort[k-1]
        after_id = asort[k+1]
        reference_epoch = py4dgeo.read_from_las(pc_list[ref_id])
        before_epoch = py4dgeo.read_from_las(pc_list[before_id])
        after_epoch = py4dgeo.read_from_las(pc_list[after_id])
        m3c2_before = py4dgeo.m3c2.M3C2(
            epochs=(reference_epoch, before_epoch),
            corepoints=reference_epoch.cloud,
            cyl_radius=0.25,
            max_distance=10,
            corepoint_normals=np.tile([0, 0, 1], (reference_epoch.cloud.shape[0], 1)),
            registration_error = 0.0
        )
        m3c2_after = py4dgeo.m3c2.M3C2(
            epochs=(reference_epoch, after_epoch),
            corepoints=reference_epoch.cloud,
            cyl_radius=0.25,
            max_distance=10,
            corepoint_normals=np.tile([0, 0, 1], (reference_epoch.cloud.shape[0], 1)),
            registration_error = 0.0
        )
        before_distances, _ = m3c2_before.run()
        after_distances, _ = m3c2_after.run()
        date_change = np.concatenate((before_distances.reshape((-1,1)),after_distances.reshape((-1,1))), axis=1)
        distances = np.append(distances, date_change, axis=0)
        coordinates = np.append(coordinates, reference_epoch.cloud, axis=0)
        t_ = np.repeat(days_el[ref_id], before_distances.shape[0]).reshape((-1,1))
        enc_ = np.repeat(encodings[ref_id].reshape((1,10)), before_distances.shape[0], axis=0)
        t = np.append(t, t_, axis=0)
        encoding = np.append(encoding, enc_, axis=0)
    tosave = np.concatenate((t, encoding, coordinates, distances), axis=1)
    np.save(f'{save_path}/bitemporal_change_{name}.npy', tosave)


def prepare_ts_for_model(save_path, name):
    analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/{name}.zip')
    timestamps = np.array((analysis.timedeltas)) + analysis.reference_epoch.timestamp
    pts = analysis.corepoints.cloud
    distances = analysis.distances
    print(timestamps.shape, pts.shape, distances.shape)
    # valid_id = np.where(np.any(np.isnan(distances), axis=1) == False)[0]
    # pts = pts[valid_id]
    encod, t_days = encode_time_info(Time(timestamps))
    print(encod.shape, t_days.shape, pts.shape)#, valid_id.shape)
    all_coords = np.empty((0,3), dtype=np.float32)
    all_days_el = np.empty((0), dtype=np.float32)
    all_encodings = np.empty((0,10), dtype=np.float32)

    for k in range(timestamps.shape[0]):
        days_el = np.repeat(t_days[k],pts.shape[0])
        encoding = np.reshape(encod[k], (1,10))
        encodings = np.repeat(encoding, pts.shape[0], axis=0)
        all_days_el = np.append(all_days_el, days_el, axis=0)
        all_encodings = np.append(all_encodings, encodings, axis=0)
        xy = pts[:,[0,1]]
        z = distances[:,k].reshape((-1,1))
        coords = np.concatenate((xy,z), axis=1)
        all_coords = np.append(all_coords, coords, axis=0)
    txyz_eval = np.concatenate((all_days_el.reshape((-1,1)), all_encodings, all_coords), axis=-1)
    print(txyz_eval.shape)
    np.save(f'{save_path}/{name}_timeseries.npy', txyz_eval)


def main():
    # data_path = '/home/mletard/compute/4dinr/data'
    # core_path = '/Users/mletard/Desktop/seasonal_grid_25cm.las'
    save_path = '/Users/mletard/Desktop'
    data_path = '/Users/mletard/Desktop/seasonal_beach'
    save_path = '/home/mletard/compute/4dinr/data'
    # pair_m3c2(data_path, save_path, "seasonal_beach")
    # gt_elevation_ts(data_path, save_path, core_path, "seasonal_beach", True)
    prepare_ts_for_model(save_path, "seasonal_beach")


if __name__ == "__main__":
    main()