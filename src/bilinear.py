import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import pandas as pd
from numba import jit


def argwhere_close(a, b, tol=1e-5):
    return np.where(np.any(np.abs(a - b[:, None]) < tol, axis=0))[0]


def argwhere_close2(a, b, tol=1e-4):
    results = []
    for el in tqdm(b):
        idx = np.abs(a - el) < tol
        results.append(idx)
    features = np.any(results, axis=0)
    return features


@jit(nopython=True)
def numba_argwhere(a, b, tol):
    n = a.shape[0]
    results = np.zeros((n, b.shape[0]), dtype=bool)
    results = []
    for j, el in enumerate(b):
        for i in range(n):
            results.append(np.abs(a - el) < tol)
    features = results.any(axis=0)
    return features


def read_test_set(data, file, n):
    test_times = np.load(file)
    idx_test = np.where(np.isin(data[:, 4], test_times))[0]
    features = np.zeros(n, dtype=bool)
    features[idx_test] = True
    return features


def load_data_h(path, test_set):
    data = np.load(path)
    idx_test = read_test_set(data, test_set, data.shape[0])
    TLatLon_train = data[~idx_test, 0:3]
    TLatLon_test = data[idx_test, 0:3]
    Z_train = data[~idx_test, 3]
    Z_test = data[idx_test, 3]
    swathid_train = data[~idx_test, 4]
    swathid_test = data[idx_test, 4]
    return TLatLon_train, Z_train, TLatLon_test, Z_test, swathid_train, swathid_test


def evaluate_model(support, z, new_locations, delta=15, downsample=1):
    print("Evaluating model...")
    z_hat = np.zeros(new_locations.shape[0])
    time_points = np.unique(new_locations[:, 0])
    for t in tqdm(list(time_points)):
        idx = (t - delta < support[:, 0]) & (support[:, 0] < t + delta)
        idx_new = new_locations[:, 0] == t
        if np.sum(idx) == 0:
            z_hat[idx_new] = np.nan
        else:
            z_hat[idx_new] = griddata(
                support[:, 1:][idx][::downsample],
                z[idx][::downsample],
                new_locations[:, 1:][idx_new],
                method="linear",
            )
    return z_hat

def load_data_faster(opt = "default"):
    if opt == 'default':
        filename = "data/data_simu.npy"
        return np.load(filename), None
    elif opt == "seasonal":
        filename = "data/seasonal.npy"
        filename_index = "data/seasonal_split.npy"
    elif opt == "daily":
        filename = "data/daily.npy"
        filename_index = "data/daily_split.npy"
    elif opt == "seasonal":
        filename = "data/monthly.npy"
        filename_index = "data//monthly_split.npy"
    return np.load(filename), np.load(filename_index)

def load_data_faster2(filename):
    filename = "data/" + filename
    return np.load(filename), np.load(filename.replace(".npy", "_split.npy"))

def split_test(data, index):
    if index is None:
        return data, None, None
    idx = (index == 2).squeeze()
    data_test = data[idx, :]
    data_train = data[~idx, :]
    return data_train, data_test, index[~idx]

def datasets():

    pairs = [
        ("daily_beach_spatial.npy", 7, 10),
        ("daily_beach_temporal.npy", 7, 10),
        ("seasonal_beach_spatial.npy", 90, 10),
        ("seasonal_beach_temporal.npy", 90, 10),
        ("monthly_beach_spatial.npy", 30, 10),
        ("monthly_beach_temporal.npy", 30, 10),
        ("weekly_beach_spatial.npy", 7, 10),
        ("weekly_beach_temporal.npy", 7, 10),
        
    ]  # (filename, delta, downsample)

    for name, delta, ds in pairs:
        
        data, index = load_data_faster2(name)

        data_train, data_test, idx = split_test(data, index)
        s_train, z_train, s_test, z_test = data_train[:, :3], data_train[:, 3], data_test[:, :3], data_test[:, 3]

        yield s_train, z_train, s_test, z_test, delta, ds, name.replace(".npy", "")


def main():

    results = pd.DataFrame(columns=["MAE", "MED", "STD"])
    for s_train, z_train, s_test, z_test, delta, ds, name in datasets():
        z_hat = evaluate_model(s_train, z_train, s_test, delta=delta, downsample=ds)
        mae = np.nanmean(np.abs(z_hat - z_test))
        med = np.nanmedian(z_hat - z_test)
        std = np.nanstd(z_hat - z_test)
        results.loc[name] = [mae, med, std]
    results.to_csv(f"bilinear_interpolation_results.csv")


if __name__ == "__main__":
    main()
