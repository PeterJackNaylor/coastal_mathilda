import numpy as np
import torch
import pinns
import py4dgeo


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
    elif opt == "seasonal_beach_temporal2":
        filename = path+"/seasonal_beach_temporal2.npy"
        filename_index = path+"/seasonal_beach_temporal2_split.npy"
    elif opt == "seasonal_beach_temporal3":
        filename = path+"/seasonal_beach_temporal3.npy"
        filename_index = path+"/seasonal_beach_temporal3_split.npy"
    elif opt == "seasonal_augmented":
        filename = path+"/seasonal_augmented.npy"
        filename_index = path+"/seasonal_augmented_split.npy"
    elif opt == "seasonal_beach_spatial2":
        filename = path+"/seasonal_beach_spatial2.npy"
        filename_index = path+"/seasonal_beach_spatial2_split.npy"
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
    if "seasonal" in opt:
        change_data = path+"/bitemporal_change_seasonal_beach.npy"
        time_series = path+"/seasonal_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/seasonal_beach.zip")
        uncertainty_data = path+"/seasonal_beach_allZ.npy"
        uncertainty_times = path+"/seasonal_beach_allT.npy"
        zmean = path+"/seasonal_beach_meanZ.npy"
        zstd = path+"/seasonal_beach_stdZ.npy"
        tmean = path+"/seasonal_beach_meanT.npy"
    elif "monthly" in opt:
        change_data = path+"/bitemporal_change_monthly_beach.npy"
        time_series = path+"/monthly_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/monthly_beach.zip")
        uncertainty_data = path+"/monthly_beach_allZ.npy"
        uncertainty_times = path+"/monthly_beach_allT.npy"
        zmean = path+"/monthly_beach_meanZ.npy"
        zstd = path+"/monthly_beach_stdZ.npy"
        tmean = path+"/monthly_beach_meanT.npy"
    elif "weekly" in opt:
        change_data = path+"/bitemporal_change_weekly_beach.npy"
        time_series = path+"/weekly_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/weekly_beach.zip")
        uncertainty_data = path+"/weekly_beach_allZ.npy"
        uncertainty_times = path+"/weekly_beach_allT.npy"
        zmean = path+"/weekly_beach_meanZ.npy"
        zstd = path+"/weekly_beach_stdZ.npy"
        tmean = path+"/weekly_beach_meanT.npy"
    elif "daily" in opt:
        change_data = path+"/bitemporal_change_daily_beach.npy"
        time_series = path+"/daily_beach_timeseries.npy"
        time_series_gt = py4dgeo.SpatiotemporalAnalysis(path+"/daily_beach.zip")
        uncertainty_data = path+"/daily_beach_allZ.npy"
        uncertainty_times = path+"/daily_beach_allT.npy"
        zmean = path+"/daily_beach_meanZ.npy"
        zstd = path+"/daily_beach_stdZ.npy"
        tmean = path+"/daily_beach_meanT.npy"

    return np.load(change_data), np.load(time_series), time_series_gt, np.load(uncertainty_data), np.load(uncertainty_times, allow_pickle=True), np.load(zmean), np.load(zstd), np.load(tmean)


def split_train(data, index):
    if index is None:
        n_data = data.shape[0]
        idx = np.arange(n_data)
        np.random.shuffle(idx)
        n_train = int(0.8 * n_data)
        idx_train = idx[:n_train]
        idx_test = idx[n_train:]
        return idx_train, idx_test
    idx = (index == 1).squeeze()
    idx_train = np.where(~idx)[0]
    idx_test = np.where(idx)[0]
    return idx_train, idx_test


def generate_single_dataloader(
    hp, data, gpu, encoding=False, nv_samples=None, nv_targets=None, train=True
):
    if encoding:
        samples = data[:, 1:-1]
        targets = data[:, -1:]
    else:
        samples = data[:, [0,-3,-2]]
        targets = data[:, -1:]

    data = TLaLoZC(
        samples,
        targets=targets,
        nv_samples=nv_samples,
        nv_targets=nv_targets,
        gpu=gpu,
        test=not train,
        hp=hp,
    )
    return data


def return_dataset(hp, data, gpu, index, encoding):
    idx_train, idx_val = split_train(
        data,
        index,
    )
    data_train = generate_single_dataloader(
        hp, data[idx_train], gpu, encoding=encoding, nv_samples=None, nv_targets=None, train=True
    )
    data_val = generate_single_dataloader(
        hp,
        data[idx_val],
        gpu,
        encoding=encoding,
        nv_samples=data_train.nv_samples,
        nv_targets=data_train.nv_targets,
        train=False,
    )
    return data_train, data_val


class dtypedData(pinns.DataPlaceholder):
    def setup_cuda(self, gpu):
        dtype = torch.float32
        if gpu:
            device = "cuda"
        else:
            device = "cpu"

        self.samples = self.samples.to(device, dtype=dtype)
        if self.need_target:
            self.targets = self.targets.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype


class TLaLoZC(dtypedData):
    # [t, lat, lon, z, swath_id, coherence]
    def __init__(
        self,
        samples,
        targets=None,
        nv_samples=None,
        nv_targets=None,
        gpu=True,
        test=True,
        hp=None,
    ):
        self.hp = hp
        self.test = test
        self.need_target = targets is not None
        self.input_size = samples.shape[1]
        self.output_size = 1
        normalise_last = True
        self.bs = hp.losses["mse"]["bs"]
        normalise_targets = hp.normalise_targets
        samples = samples.astype(np.float32)
        if self.need_target:
            targets = targets.astype(np.float32)

        nv_samples = self.normalize(samples, nv_samples, normalise_last)
        if self.need_target:
            if not normalise_targets:
                nv_targets = [(0, 1) for _ in range(targets.shape[1])]
            nv_targets = self.normalize(targets, nv_targets, True)

        self.samples = torch.from_numpy(samples).float()
        self.nv_samples = nv_samples
        self.nv_targets = nv_targets

        if self.need_target:
            self.targets = torch.from_numpy(targets)

        self.setup_cuda(gpu)
        self.setup_batch_idx()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.need_target:
            # return sample
            return {"x": sample}
        target = self.targets[idx]
        output = {"x": sample, "z": target}
        return output
