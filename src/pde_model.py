import pinns
import torch 

def spatial_temporal_grad(model, t, Lat, Lon):
    torch.set_grad_enabled(True)
    Lat.requires_grad_(True)
    Lon.requires_grad_(True)
    t.requires_grad_(True)
    u = model(t, Lat, Lon)
    du_dLat = pinns.gradient(u, Lat)
    du_dLon = pinns.gradient(u, Lon)
    du_dt = pinns.gradient(u, t)
    return du_dLat, du_dLon, du_dt




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
        Lat = pinns.gen_uniform(self.hp.losses["spatial_grad"]["bs"], self.device)
        Lon = pinns.gen_uniform(self.hp.losses["spatial_grad"]["bs"], self.device)

        M = self.M if hasattr(self, "M") else None
        temporal_scheme = self.hp.losses["spatial_grad"]["temporal_causality"]

        t = pinns.gen_uniform(
            self.hp.losses["spatial_grad"]["bs"],
            self.device,
            start=-1,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
            dtype=self.dtype,
        )
        grad = spatial_grad(self.model, t, Lat, Lon)
        return grad
