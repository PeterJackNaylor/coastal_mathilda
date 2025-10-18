import numpy as np
import laspy
import os
import argparse
from datetime import datetime
from astropy.time import Time
from temp_encoding import encode_time_info
import open3d as o3d
from shutil import copy 
import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_yaml(path):
    with open(path) as f:
        params = yaml.load(f, Loader=yaml.Loader)
    return AttrDict(params)


def parser_f(actions):
    parser = argparse.ArgumentParser(
        description="Preparing data for INR",
    )
    parser.add_argument("--name", type=str, help="Name given to saved files")
    parser.add_argument("--mode", choices=actions.keys(), required=True)
    parser.add_argument(
        "--yaml_file",
        type=str,
        default="data_config.yml",
        help="Configuration yaml file for the dataset characteristics",
    )
    # parser.add_argument(
    #     "--keyword",
    #     type=str,
    #     help="keyword: seasonal, default",
    # )
    args = parser.parse_args()
    return args


def setup_p(
    yaml_params,
    name,
):
    params = read_yaml(yaml_params)
    copy(yaml_params, f"{params.path}/data_config_{name}.yml")
    return params


def get_files_times(path_las, file_id = ".las"):
    files = []
    dates = np.empty((0))
    dates_str = np.empty((0))
    for r, d, f in os.walk(path_las):
            for file in f:
                if file_id in file:
                    joined = os.path.join(path_las, file)
                    files.append(joined)
                    timestamp_str = '_'.join(file.split('.')[0].split('_')[:]) # yields YYMMDD_hhmmss
                    timestamp = np.array(([datetime.strptime(timestamp_str, '%y%m%d_%H%M%S')]))
                    dates_str = np.append(dates_str, np.array(([timestamp_str])), axis=0)
                    dates = np.append(dates, timestamp, axis=0)
    return files, Time(dates), dates_str


def get_txyz(las_file, t_days, t_enc):
    data = laspy.read(las_file)
    xyz = data.xyz
    npts = data.x.shape[0]
    encodings = np.repeat(t_enc.reshape((1,10)), npts, axis=0)
    days_el = np.repeat(t_days.reshape((1,1)), npts, axis=0)
    return np.concatenate((days_el, encodings, xyz), axis=-1)


def temporal_sub(txyz, voxel_size):
    print("TEMPORAL")
    return np.empty((0,txyz.shape[1])), txyz


def space_sub(txyz, voxel_size):
    print("SPATIAL")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(txyz[...,-3:])
    down = pc.voxel_down_sample(voxel_size)
    down_xyz = np.asarray(down.points)
    t_ = txyz[:down_xyz.shape[0],:-3]
    return np.concatenate((t_, down_xyz), axis=-1), txyz


def random_sub(txyz, voxel_size):
    print("RANDOM")
    n_val = int(0.4 * txyz.shape[0])
    ival = np.random.choice(txyz.shape[0], size=n_val, replace=False)
    train_mask = np.ones(txyz.shape[0], bool)
    train_mask[ival] = False
    train_part = txyz[train_mask]
    val_part = txyz[ival]
    return train_part, val_part

    
def save_to_npy(train, val, test, name, suffix=''):
    idtrain = np.zeros((train.shape[0],1), dtype=np.uint8)
    idval = np.ones((val.shape[0],1), dtype=np.uint8)
    idtest = np.ones((test.shape[0],1), dtype=np.uint8) * 2
    txyz_all = np.concatenate((train, val, test), axis=0)
    split_id = np.concatenate((idtrain, idval, idtest), axis=0)
    print(txyz_all.shape, split_id.shape)
    np.save(name+suffix+".npy", txyz_all)#.astype(np.float32))
    np.save(name+suffix+"_split.npy", split_id)#.astype(np.float32))
    return txyz_all, split_id


def get_dates(dates_str):
    dates = np.empty((0))
    for d in dates_str:
        date = np.array(([datetime.strptime(d, '%y%m%d_%H%M%S')]))
        dates = np.append(dates, date, axis=0)
    return Time(dates)

def main():
    actions = {
        "temporal": temporal_sub,
        "random": random_sub,
        "spatial": space_sub
    }
    opt = parser_f(actions)
    data_p = setup_p(
        opt.yaml_file,
        opt.name,
    )
    sub_fn = actions[opt.mode]
    dates_val = data_p["dates_val"]
    dates_test = data_p["dates_test"]
    print(dates_val)
    print(dates_test)
    # import pdb; pdb.set_trace()
    files, dates, dates_str = get_files_times(data_p["path"])
    t_enc, t_days = encode_time_info(dates)
    print(t_days.shape, t_enc.shape)
    print(len(files), dates.shape, len(dates_str))
    train = np.empty((0, t_enc.shape[1]+4))
    val = np.empty((0, t_enc.shape[1]+4))
    test = np.empty((0, t_enc.shape[1]+4))
    for k in range(len(files)):
        txyz = get_txyz(files[k], t_days[k], t_enc[k])
        print(txyz.shape, dates_str[k], dates[k])
        if dates_str[k] in dates_val:
            print("VAL ", dates_str[k], dates_val)
            train_part, val_part = sub_fn(txyz, data_p["voxel_size"])
            train = np.append(train, train_part, axis=0)
            val = np.append(val, val_part, axis=0)
        elif dates_str[k] in dates_test:
            train_part, test_part = sub_fn(txyz, data_p["voxel_size"])
            train = np.append(train, train_part, axis=0)
            test = np.append(test, test_part, axis=0)
        else:
            train = np.append(train, txyz, axis=0)
    txyz_all, split_id = save_to_npy(train, val, test, data_p["path"]+"/"+opt.name, "")
    print('\n---------------- DATA INFORMATION ----------------')
    print("NUMBER OF ACQUISITIONS:", len(files))
    print("TOTAL NB OF POINTS: ", txyz_all.shape[0])
    print("NUMBER OF TRAIN SAMPLES: ", train.shape[0])
    print("NUMBER OF VAL SAMPLES: ", val.shape[0])
    print("NUMBER OF TEST SAMPLES: ", test.shape[0])
    print('--------------------------')
    print("\nShape of TXYZ train: ", train.shape)
    print("Shape of TXYZ val: ", val.shape)
    print("Shape of TXYZ test: ", test.shape)
    print("Dates val: ", dates_val)
    print("Dates test: ", dates_test)
    print('----------------------------------------------------')


if __name__ == "__main__":
    main()