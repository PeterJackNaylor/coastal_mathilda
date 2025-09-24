import numpy as np
import py4dgeo
import os
from datetime import datetime
import py4dgeo


# specify the data path
data_path = '/home/mletard/data/4d_training/beach_data/02_data_extracted'
save_path = '/home/mletard/data/4d_training/beach_data'

##### BELOW, CREATION OF THE ANALYSIS OBJECT #################################

def create_analysis(data_path, save_path, name):
    analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/{name}.zip')
    # check if the specified path exists
    if not os.path.isdir(save_path):
        print(f'ERROR: {save_path} does not exist')
        print('Please specify the correct path to the data directory by replacing <save_path> above.')

    # list of point clouds (time series)
    pc_list = []
    pc_folders_list = os.listdir(data_path)

    for f in pc_folders_list:
        # sub-directory containing the point clouds
        pc_dir = os.path.join(os.path.join(data_path, f),f)
        files = os.listdir(pc_dir)
        for ff in files:
            if ".laz" in ff or ".las" in ff:
                pc_list.append(os.path.join(pc_dir, ff))
    print(len(pc_list), " point clouds.")

    # read the timestamps from file names
    timestamps = []
    for f in pc_list:
        # get the timestamp from the file name   
        timestamp_str = '_'.join(f.split('.')[0].split('/')[-1:]) # yields YYMMDD_hhmmss
        # convert string to datetime object
        timestamp = datetime.strptime(timestamp_str, '%y%m%d_%H%M%S')
        timestamps.append(timestamp)

    print(timestamps[:5])
    print(pc_list[:2])

    analysis = py4dgeo.SpatiotemporalAnalysis(f'{save_path}/beach.zip', force=True)

    # specify the reference epoch
    reference_epoch_file = pc_list[0]

    # read the reference epoch and set the timestamp
    reference_epoch = py4dgeo.read_from_las(reference_epoch_file)
    reference_epoch.timestamp = timestamps[0]

    # set the reference epoch in the spatiotemporal analysis object
    analysis.reference_epoch = reference_epoch

    # specify corepoints, here every 10 points of the reference epoch
    analysis.corepoints = reference_epoch.cloud[::]

    # specify M3C2 parameters
    analysis.m3c2 = py4dgeo.M3C2(cyl_radii=(1.0,), normal_radii=(1.0,), max_distance=10.0, registration_error = 0.025)

    # create a list to collect epoch objects
    epochs = []
    for e, pc_file in enumerate(pc_list[1:]):  
        epoch = py4dgeo.read_from_las(pc_file)
        epoch.timestamp = timestamps[e]
        epochs.append(epoch)

    # add epoch objects to the spatiotemporal analysis object
    analysis.add_epochs(*epochs)

    print(f"Space-time distance array:\n{analysis.distances[:3,:5]}")
    print(f"Uncertainties of M3C2 distance calculation:\n{analysis.uncertainties['lodetection'][:3, :5]}")
    print(f"Timestamp deltas of analysis:\n{analysis.timedeltas[:5]}")




def pair_m3c2(path):
    ori_m3c2_run = py4dgeo.m3c2.M3C2(
        epochs=(reference_epoch, target_epoch),
        corepoints=corepoints,
        cyl_radius=0.25,
        max_distance=10,
        corepoint_normals=np.tile([0, 0, 1], (corepoints.shape[0], 1))
    )
    ori_distances, _ = ori_m3c2_run.run()
    ori_nan_percentage = np.isnan(ori_distances).sum() / len(ori_distances) * 100 if len(ori_distances) > 0 else 100