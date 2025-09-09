import numpy as np
import torch
import pinns

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
        samples = data[:, :12]
        targets = data[:, 12:13]
    else:
        samples = data[:, :3]
        targets = data[:, 3:4]

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
