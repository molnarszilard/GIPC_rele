import torch.utils.data as data
import numpy as np
from pathlib import Path
import torch
import cv2
import os
class DatasetLoader(data.Dataset):
    
    def __init__(self, root='./data/ModelNet10_gim_64/', seed=None, train=True):
        np.random.seed(seed)
        self.root = Path(root)

        if train:
            self.depth_input_paths = [root+'/train/'+d for d in os.listdir(root+'/train/')]
        else:
            self.depth_input_paths = [root+'/test/'+d for d in os.listdir(root+'/test/')]
        
        self.length = len(self.depth_input_paths)
            
    def __getitem__(self, index):
        pathgim = self.depth_input_paths[index]      
        gimgt=cv2.imread(pathgim,cv2.IMREAD_UNCHANGED).astype(np.float32)
        gimgt=torch.from_numpy(np.moveaxis(gimgt,-1,0))
        gimgt=gimgt/255
        gimgt_rgb=gimgt*0
        gimgt_rgb[0]=gimgt[1]
        gimgt_rgb[1]=gimgt[0]
        gimgt_rgb[2]=gimgt[2]
        return gimgt_rgb

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DatasetLoader()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
