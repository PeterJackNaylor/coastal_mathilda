import numpy as np
import py4dgeo
import os
from datetime import datetime
import py4dgeo
import matplotlib.pyplot as plt


def gt_elevation_ts(data_path, save_path, corepoints_path, name, make_analysis = False):
    analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/{name}.zip')
    if make_analysis:
        core_pc = py4dgeo.read_from_las(corepoints_path)
        corepoints = core_pc.cloud
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
        ref_id = asort[0]
        analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/{name}.zip', force=True)
        reference_epoch_file = pc_list[ref_id]
        reference_epoch = py4dgeo.read_from_las(reference_epoch_file)
        reference_epoch.timestamp = timestamps[ref_id]
        analysis.reference_epoch = reference_epoch
        analysis.corepoints = corepoints

        # specify M3C2 parameters
        analysis.m3c2 = py4dgeo.M3C2(cyl_radius=0.25, corepoint_normals=np.tile([0, 0, 1], (corepoints.shape[0], 1)), max_distance=10.0, registration_error = 0.025)
        # create a list to collect epoch objects
        epochs = []
        for k in range(1,asort.shape[0],1):
            pc_id = asort[k]
            epoch = py4dgeo.read_from_las(pc_list[pc_id])
            epoch.timestamp = timestamps[pc_id]
            epochs.append(epoch)
        analysis.add_epochs(*epochs)

        corept_dummy = corepoints
        print(corepoints.shape)
        print(stop)

        print(f"Space-time distance array:\n{analysis.distances[:3,:5]}")
        print(f"Uncertainties of M3C2 distance calculation:\n{analysis.uncertainties['lodetection'][:3, :5]}")
        print(f"Timestamp deltas of analysis:\n{analysis.timedeltas[:5]}")
        print(np.all(np.isnan(analysis.distances)))

    fig=plt.figure(figsize=(12,5))
    ax1=fig.add_subplot(1,2,1,projection='3d',computed_zorder=False)
    ax2=fig.add_subplot(1,2,2)
    # get the corepoints
    corepoints = analysis.corepoints.cloud
    # get change values of last epoch for all corepoints
    distances = analysis.distances
    distances_epoch = [d[2] for d in distances]
    # get the time series of changes at a specific core point locations
    cp_idx_sel = 54660
    coord_sel = analysis.corepoints.cloud[cp_idx_sel]
    timeseries_sel = distances[cp_idx_sel]
    print(np.where(np.any(np.isnan(distances), axis=1) == False)[0])
    print(timeseries_sel)
    # get the list of timestamps from the reference epoch timestamp and timedeltas
    timestamps = [t + analysis.reference_epoch.timestamp for t in analysis.timedeltas]
    # plot the scene
    d = ax1.scatter(corepoints[:,0], corepoints[:,1], corepoints[:,2], c=distances_epoch[:], cmap='seismic_r', vmin=-1.5, vmax=1.5, s=1, zorder=1) 
    plt.colorbar(d, format=('%.2f'), label='Distance [m]', ax=ax1, shrink=.5, pad=.15)
    # add the location of the selected coordinate
    ax1.scatter(coord_sel[0], coord_sel[1], coord_sel[2], c='black', s=3, zorder=2, label='Selected location')
    ax1.legend()
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_aspect('equal')
    ax1.view_init(elev=30., azim=150.)
    ax1.set_title('Changes at %s' % (analysis.reference_epoch.timestamp+analysis.timedeltas[2]))
    # plot the time series
    ax2.plot(timestamps, timeseries_sel, color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Distance [m]')
    ax2.grid()
    ax2.set_ylim(-0.2,1.0)
    ax2.set_title('Time series at selected location')
    plt.tight_layout()
    plt.show()
    plt.savefig('test.png')




def pair_m3c2(data_path, corepoints_path):
    core_pc = py4dgeo.read_from_las(corepoints_path)
    corepoints = core_pc.cloud
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
    distances = np.empty((corepoints.shape[0], 0))
    for k in range(1,asort.shape[0],1):
        ref_id = asort[k-1]
        targ_id = asort[k]
        reference_epoch = py4dgeo.read_from_las(pc_list[ref_id])
        target_epoch = py4dgeo.read_from_las(pc_list[targ_id])
    
        ori_m3c2_run = py4dgeo.m3c2.M3C2(
            epochs=(reference_epoch, target_epoch),
            corepoints=corepoints,
            cyl_radius=0.25,
            max_distance=10,
            corepoint_normals=np.tile([0, 0, 1], (corepoints.shape[0], 1))
        )
        ori_distances, _ = ori_m3c2_run.run()
        print(ori_distances.shape)
        distances = np.append(distances, ori_distances.reshape((-1,1)), axis=1)
        print(distances.shape)

def main():
    data_path = '/Users/mletard/Desktop/monthly_beach'
    core_path = '/Users/mletard/Desktop/monthly_merged_10cm.las'
    save_path = '/Users/mletard/Desktop'
    # pair_m3c2(data_path, core_path)
    gt_elevation_ts(data_path, save_path, core_path, "monthly_beach", True)


if __name__ == "__main__":
    main()