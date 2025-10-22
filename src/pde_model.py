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
        grad = spatial_temporal_grad(self.model, t, Lat, Lon)
        return grad
    

    def compute_grads(self):
        if not hasattr(self, "it_comp"):
            self.it_comp = 0
        if self.it != self.it_comp:
            try:
                bs = self.hp.losses["gradient_lat"]["bs"]
            except:
                bs = self.hp.losses["temporal_grad"]["bs"]
            Lat = pinns.gen_uniform(bs, self.device)
            Lon = pinns.gen_uniform(bs, self.device)
            M = self.M if hasattr(self, "M") else None
            try:
                temporal_scheme = self.hp.losses["gradient_lat"]["temporal_causality"]
            except:
                temporal_scheme = self.hp.losses["temporal_grad"]["temporal_causality"]

            t = pinns.gen_uniform(
                bs,
                self.device,
                # start=0,
                # end=1,
                temporal_scheme=temporal_scheme,
                M=M,
            )
            # if self.need_hessian:
            #     (
            #         grad_lat,
            #         grad_lon,
            #         grad_t,
            #         grad_lat2,
            #         grad_lon2,
            #         grad_lonlat,
            #     ) = spatial_temporal_grad(self.model, t, Lat, Lon, True)
            #     self.grad_lon2 = grad_lat2
            #     self.grad_lat2 = grad_lon2
            #     self.grad_lonlat = grad_lonlat
            # else:
            grad_lat, grad_lon, grad_t = spatial_temporal_grad(
                self.model, t, Lat, Lon
            )

            self.grad_lat = grad_lat
            self.grad_lon = grad_lon
            self.grad_t = grad_t
            self.it_comp = self.it

    def gradient_lat(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_lat / self.data.nv_samples[1][1]

    def gradient_lon(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_lon / self.data.nv_samples[2][1]

    def gradient_time(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_t / self.data.nv_samples[0][1]
