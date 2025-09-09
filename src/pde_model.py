import pinns
import torch 

class Surface(pinns.DensityEstimator):
    def autocasting(self):
        if self.device == "cpu":
            dtype = torch.bfloat16
            if self.hp.model["name"] == "WIRES":
                dtype = torch.bfloat32
        else:
            dtype = torch.float16
            if self.hp.model["name"] == "WIRES":
                dtype = torch.float32
        self.use_amp = True
        if self.hp.model["name"] == "WIRES":
            self.use_amp = False
        self.dtype = dtype

    def spatial_gradient(self, z, z_hat, weight):
        x = pinns.gen_uniform(self.hp.losses["spatial_grad"]["bs"], self.device)

        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses["spatial_grad"]["temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses["spatial_grad"]["bs"],
            self.device,
            start=0,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
            dtype=self.dtype,
        )
        grad = spatial_grad(self.model, x, t)
        return grad
