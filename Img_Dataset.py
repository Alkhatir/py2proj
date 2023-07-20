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


class Imgdataset(Dataset):
    def __init__(self, image_dir: str) -> None:
        super().__init__()
        self.imgs = sorted(osp.abspath(f) for f in glob(
            osp.join(image_dir, "**", "*.jpg"), recursive=True))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        im_shape = 64
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape, interpolation=Image.BILINEAR),
            transforms.CenterCrop(size=(im_shape, im_shape)),
        ])
        with Image.open(self.imgs[index]) as tmp:
            image = resize_transforms(tmp)
            image = to_grayscale(np.array(image))
        x = rd.randint(1, rd.randint(4, 32))
        pixelated_image, known_array, target_array, target_array_with_padding = prepare_image(
            image, x=x, y=x, width=rd.randint(4, 32), height=rd.randint(4, 32), size=rd.randint(4, 16))
        return torch.from_numpy(pixelated_image), torch.from_numpy(known_array), torch.from_numpy(target_array), torch.from_numpy(target_array_with_padding)


if __name__ == "__main__":
    dataset = Imgdataset(
        r"C:\Users\peerh\OneDrive\Desktop\py2proj\training\000")
    for pixelated_image, known_array, target_array,target_array_with_padding in dataset:
        torch.set_printoptions(profile="full")
        #print((~known_array)*1)
        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(pixelated_image[0].numpy(), cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("pixelated_image")
        axes[1].imshow(known_array[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("known_array")
        axes[2].imshow(target_array[0].numpy(), cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("target_array")
        fig.suptitle("000")
        fig.tight_layout()
        plt.show()
