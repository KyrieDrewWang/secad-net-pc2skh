import numpy as np
import os
import torch
import torch.utils.data
import h5py
import random
import h5py

def readIndex(index_path, shuffle=False):
    f_lst = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip()
            f_lst.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(f_lst)
    return f_lst

class GTSamples(torch.utils.data.Dataset):
    """Dataset for training
    """
    def __init__(self,data_source,grid_sample=64, num_testing_points=4096):
        print('data source', data_source)
        self.data_urls = readIndex(data_source)
        self.num_testing_points = num_testing_points

    def __len__(self):
        return len(self.data_urls)

    def __getitem__(self, idx):
        data_url = self.data_urls[idx]
        
        _, data_id = os.path.split(data_url[0])  
        data_id = data_id.replace(".h5", "")

        with h5py.File(data_url, 'r') as fp:
            sample_voxels = fp["vox"][:].astype(np.float32)
            pointcloud = fp["pc"][:].astype(np.float32)
            id = fp["id"]
        sample_voxels = torch.from_numpy(sample_voxels).float()
        sample_voxels = sample_voxels.unsqueeze(0)
        
        testing_indices = np.random.randint(0, pointcloud.shape[0], self.num_testing_points)
        pointcloud = pointcloud[testing_indices]       
        points = torch.from_numpy(pointcloud).float()
        
        return {"voxels":sample_voxels, "occ_data": points}


# class VoxelSamples(torch.utils.data.Dataset):
# 	"""Dataset for fine-tuning and testing
# 	"""
# 	def __init__(
# 		self,
# 		data_source
# 	):
# 		print('data source', data_source)
# 		self.data_source = data_source
# 		print('class Samples from voxels')

# 		name_file = os.path.join(self.data_source, 'test_names.npz')
# 		npz_shapes = np.load(name_file)
# 		self.data_names = npz_shapes['test_names']
  
# 		filename_voxels = os.path.join(self.data_source, 'voxel2mesh.hdf5')
# 		data_dict = h5py.File(filename_voxels, 'r')
# 		data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()

# 		self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
# 		self.data_points = torch.from_numpy(data_dict['points'][:]).float()
# 		self.data_points[:, :, :3] = (self.data_points[:, :, :3] + 0.5)/64-0.5
# 		data_dict.close()
			
# 		print('Loaded voxels shape, ', self.data_voxels.shape)
# 		print('Loaded points shape, ', self.data_points.shape)


# 	def __len__(self):
# 		return len(self.data_voxels)

# 	def __getitem__(self, idx):
# 		return self.data_voxels[idx], self.data_points[idx]

class VoxelSamples(torch.utils.data.Dataset):
    """Dataset for fine-tuning and testing
    """
    def __init__(
        self,
        data_source,
        num_testing_points = 4096
    ):
        self.data_urls = readIndex(data_source)
        self.num_testing_points = num_testing_points

    def __len__(self):
        return len(self.data_voxels)

    def __getitem__(self, idx):
        data_url = self.data_urls[idx]
        
        _, data_id = os.path.split(data_url)  
        data_id = data_id.replace(".h5", "")

        with h5py.File(data_url, 'r') as fp:
            sample_voxels = fp["vox"][:].astype(np.float32)
            pointcloud = fp["pc"][:].astype(np.float32)
            id = fp["id"]
        sample_voxels = torch.from_numpy(sample_voxels).float()
        sample_voxels = sample_voxels.unsqueeze(0)
        
        testing_indices = np.random.randint(0, pointcloud.shape[0], self.num_testing_points)
        pointcloud = pointcloud[testing_indices]       
        points = torch.from_numpy(pointcloud).float()
        
        return sample_voxels, points, data_id

if __name__ == "__main__":
    # data = VoxelSamples("data/abc_all")
    # print(data[1])
    
    data = GTSamples("/data/wc/SECAD-Net/data/secad/data_train.txt")
    
    print(data[1])