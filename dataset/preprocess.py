from dataloader import readIndex
import os
import numpy as np
import math
import h5py
from tqdm import tqdm
from multiprocessing import Pool
import json
import trimesh
import random
from plyfile import PlyData, PlyElement
from triangle_hash import TriangleHash as _TriangleHash

def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

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

NUM_POINTS_UNIFORM = 8192

def readIndex(index_path, shuffle=False):
    f_lst = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            f_lst.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(f_lst)
    return f_lst

class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = np.zeros(points.shape[0], dtype=bool)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        
        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains

class MeshIntersector:
    def __init__(self, mesh, resolution=512):
        triangles = mesh.vertices[mesh.faces].astype(np.float64)
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles)
        # assert(np.allclose(triangles.reshape(-1, 3).min(0), 0.5))
        # assert(np.allclose(triangles.reshape(-1, 3).max(0), resolution - 0.5))

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        contains = np.zeros(len(points), dtype=bool)

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])
        
        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        if (contains1 != contains2).any():
            print('Warning: contains1 != contains2 for some points.')
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)
    
        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array

def check_mesh_contains(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    return contains


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
    # origin = pts.mean(axis=0)
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
    # with open(os.path.join(output_folder, object_name + "_voxels.obj"), "w") as fout:
    #     for p in points:fout.write("v " + " ".join(map(str, p)) + "\n")
    #     for f in faces+1:fout.write("f " + " ".join(map(str, f)) + "\n")
    # print(object_name, "done.")

    mesh = trimesh.Trimesh(vertices = points, faces = faces)
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1])/2
    scale = (bbox[1] - bbox[0])
    points_uniform = (np.random.rand(NUM_POINTS_UNIFORM, 3) - 0.5) * scale
    occupancies = check_mesh_contains(mesh, points_uniform).astype(np.uint8)

    return grid, np.concatenate([points_uniform, np.expand_dims(occupancies, 1)], axis=1)


def process(data_url, num_testing_points=8192):
    _, data_id = os.path.split(data_url[0])  
    data_id = data_id.replace("_sdf.vox", "")
    
    mesh_path = data_url[0].replace("_sdf.vox", "_centered.obj")
    pc_path  = data_url[1]
    
    try:
        sample_voxels, pointcloud = vox(mesh_path)
    except:
        return
    testing_indices = np.random.randint(0, pointcloud.shape[0], num_testing_points)
    pointcloud = pointcloud[testing_indices]  

    # write_ply(pointcloud, os.path.join(SAVE_H5_base, data_id + '.ply'))

    SAVE_H5_PATH = os.path.join(SAVE_H5_base, data_id + '.h5')  
    with h5py.File(SAVE_H5_PATH, 'w') as fp:
        fp.create_dataset('vox', data=sample_voxels, dtype=np.float32)
        fp.create_dataset('pc', data=pointcloud, dtype=np.float32)
        fp.create_dataset('id', data=data_id)
    return

if __name__ == "__main__":
    
    data_source = "/data/wc/SECAD-Net/data/secad/train.txt"
    SAVE_H5_base = "/data/wc/SECAD-Net/data/secad_8192/h5"
    output_folder = "/data/wc/SECAD-Net/data/secad_8192"
    
    data_urls = readIndex(data_source)
    pbar = tqdm(data_urls, total=len(data_urls))
    
    P = Pool(64, maxtasksperchild=4)
    for d in pbar:
        P.apply(process, args=(d, ))
        # process(d)
    P.close()
    P.join()