from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os.path as osp
import random as rd
from a2_ex2 import prepare_image
from a2_ex1 import to_grayscale
import matplotlib.pyplot as plt
import pickle

class Testdataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open("C:\\Users\\peerh\\OneDrive\\Desktop\\py2proj\\test_set.pkl","rb") as f:
            dict = pickle.load(f)
            self.pixelated_images = dict["pixelated_images"]
            self.known_arrays = dict["known_arrays"]
    def __len__(self) -> int:
        return len(self.pixelated_images)
    def __getitem__(self, index):
        return torch.from_numpy(self.pixelated_images[index]), torch.from_numpy(self.known_arrays[index])
    
if __name__ == "__main__":
    dataset = Testdataset()
    for pixelated_image, known_array in dataset:
        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(pixelated_image[0].numpy(), cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("pixelated_image")
        axes[1].imshow(known_array[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("known_array")
        fig.suptitle("000")
        fig.tight_layout()
        plt.show()