from dataloader import readIndex
import os
from dataloader import load_vox
import numpy as np
import math
import h5py
from tqdm import tqdm
from multiprocessing import Pool
import json
import trimesh

def pathrename(obj_path, suffix):
    '''
    given the absolute path of the file, return the file name without suffix
    '''
    base_path, file_name = os.path.split(obj_path)
    file_id = file_name.split('.')[0]
    new_path = os.path.join(base_path, file_id + suffix)
    return new_path

def create_if_needed(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def vox(path, resolution=[64, 64, 64], sampling=10000):

    input_mesh_filename = path
    object_name = os.path.splitext(os.path.basename(path))[0]
    RES_X, RES_Y, RES_Z = resolution
    sample_points_count = sampling
    # create_if_needed(output_folder)
    mesh = trimesh.exchange.load.load(input_mesh_filename)
    # Uniform Points Sampling
    pts, _ = trimesh.sample.sample_surface_even(mesh, sample_points_count )
    # Save sample points
    # sampled_points_mesh = trimesh.Trimesh(vertices=pts)
    # sampled_points_mesh.export(os.path.join(output_folder, object_name + "_resampled_points.ply"))
    # Adjust the grid origin and voxels size
    origin = pts.min(axis=0)
    dimensions = pts.max(axis=0) - pts.min(axis=0)
    scales = np.divide(dimensions, np.array([RES_X-1, RES_Y-1, RES_Z-1]))
    scale = np.max(scales)
    # Voxelize
    pts -= origin
    pts /= scale
    pts_int = np.round(pts).astype(int)

    grid = np.zeros((RES_X, RES_Y, RES_Z), dtype=int)
    gooRES_X = np.where(np.logical_and(pts_int[:, 0] >= 0, pts_int[:, 0] < RES_X))[0]
    gooRES_Y = np.where(np.logical_and(pts_int[:, 1] >= 0, pts_int[:, 1] < RES_Y))[0]
    gooRES_Z = np.where(np.logical_and(pts_int[:, 2] >= 0, pts_int[:, 2] < RES_Z))[0]
    goods = np.intersect1d(np.intersect1d(gooRES_X, gooRES_Y), gooRES_Z)
    pts_int = pts_int[goods, :]
    grid[pts_int[:, 0], pts_int[:, 1], pts_int[:, 2]] = 1
    return grid
    """
    # Save voxels
    voxel_pts = np.array([[-0.5, 0.5, -0.5],
                        [0.5, 0.5, -0.5],
                        [0.5, 0.5, 0.5],
                        [-0.5, 0.5, 0.5],
                        [-0.5, -0.5, -0.5],
                        [0.5, -0.5, -0.5],
                        [0.5, -0.5, 0.5],
                        [-0.5, -0.5, 0.5]])
    voxel_faces = np.array([[0, 1, 2, 3],
                            [1, 5, 6, 2],
                            [5, 4, 7, 6],
                            [4, 0, 3, 7],
                            [0, 4, 5, 1],
                            [7, 3, 2, 6]])
    def get_voxel(i, j, k):
        voxel_pts, voxel_faces
        v = np.array([i, j, k], dtype=float) * scale
        v += origin
        points = voxel_pts * scale + v
        return points, voxel_faces.copy()
    points = []
    faces = []
    fi = 0
    for i in range(RES_X):
        for j in range(RES_Y):
            for k in range(RES_Z):
                if grid[i, j, k]:
                    p, f = get_voxel(i, j, k)
                    points.append(p)
                    f += fi
                    faces.append(f)
                    fi += 8
    points = np.vstack(points)
    faces = np.vstack(faces)
    # Write obj mesh with quad faces
    with open(os.path.join(output_folder, object_name + "_voxels.obj"), "w") as fout:
        for p in points:fout.write("v " + " ".join(map(str, p)) + "\n")
        for f in faces+1:fout.write("f " + " ".join(map(str, f)) + "\n")
    print(object_name, "done.")
    """

def process(data_url, num_testing_points=4096):
    _, data_id = os.path.split(data_url[0])  
    data_id = data_id.replace("_sdf.vox", "")
    
    mesh_path = data_url[0].replace("_sdf.vox", "_centered.obj")
    pc_path  = data_url[1]
    
    sample_voxels = vox(mesh_path)
    
    pointcloud = np.load(pc_path)
    testing_indices = np.random.randint(0, pointcloud.shape[0], num_testing_points)
    pointcloud = pointcloud[testing_indices]  
    
    SAVE_H5_PATH = os.path.join(SAVE_H5_base, data_id + '.h5')  
    with h5py.File(SAVE_H5_PATH, 'w') as fp:
        fp.create_dataset('vox', data=sample_voxels, dtype=np.float32)
        fp.create_dataset('pc', data=pointcloud, dtype=np.float32)
        fp.create_dataset('id', data=data_id)

if __name__ == "__main__":
    
    data_source = "/data/wc/SECAD-Net/data/secad/train.txt"
    SAVE_H5_base = "/data/wc/SECAD-Net/data/secad/data"
    output_folder = "/data/wc/SECAD-Net/data/secad/data"
    
    data_urls = readIndex(data_source)
    pbar = tqdm(data_urls, total=len(data_urls))
    
    P = Pool(128, maxtasksperchild=4)
    for d in pbar:
        P.apply(process, args=(d, ))
        # process(d)
    P.close()
    P.join()