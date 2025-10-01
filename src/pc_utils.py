import numpy as np
import open3d as o3d
import scipy

def get_roughness(corepts, cloud, scale):
    """
    Compute roughness of a given point cloud (residuals of LS plane fit)

    Parameters
    ----------
    corepts : array (Nx3)
        XYZ coordinates of the core points.
    cloud : array (Nx3)
        XYZ coordinates of the N points constituting the point cloud.
    scale : float
        radius for the search around each point of the cloud.

    Returns 
    -------
    roughness : array(N,1)
    """
    pts = cloud[:, [0, 1, 2]]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)        
    pc_tree = o3d.geometry.KDTreeFlann(pc)    
    corepts = corepts[:, [0, 1, 2]]
    corepc = o3d.geometry.PointCloud()
    corepc.points = o3d.utility.Vector3dVector(corepts)
    n_p = corepts.shape[0]

    roughness = np.empty((0,1), dtype=np.float32)
    dzdx = np.empty((0,1), dtype=np.float32)
    dzdy = np.empty((0,1), dtype=np.float32)
    for i in range(corepts.shape[0]):
        p = corepc.points[i]
        krad, idx_sphere, _ = pc_tree.search_radius_vector_3d(p, scale/2)
        sphere = np.asarray(pc.points)[idx_sphere, :]
        if len(idx_sphere) > 0:
            z_sphere = sphere[:, 2]
            sphere_pc = o3d.geometry.PointCloud()
            sphere_pc.points = o3d.utility.Vector3dVector(sphere)
            A = np.c_[sphere[:, 0], sphere[:, 1], np.ones(sphere.shape[0])]
            C, resid, _, _ = scipy.linalg.lstsq(A, sphere[:, 2])
            plane_residuals = np.mean(resid)
            roughness = np.append(roughness, plane_residuals.reshape((-1,1)), axis=0)
            dzdx = np.append(dzdx, C[0].reshape((-1,1)), axis=0)
            dzdy = np.append(dzdy, C[1].reshape((-1,1)), axis=0)
    return roughness, dzdx, dzdy