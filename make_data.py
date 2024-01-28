import os
import torch
from torchvision.transforms import transforms
from tqdm import tqdm


def make_imgs_and_labels(save_root):
    """
    Create two groups of random noise images with 500 images in each group.

    saveroot
        group1
            img_0.png
            ...
        group2
            img_0.png
            ...
    """

    os.makedirs(os.path.join(save_root, "group1"))
    os.makedirs(os.path.join(save_root, "group2"))

    print("Creating the first group images")
    for i in tqdm(range(500)):
        img = torch.randn((3, 128, 128))
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(save_root, "group1", f"img_{i}.png"))

    print("Creating the first group images")
    for i in tqdm(range(500)):
        img = torch.randn((3, 128, 128))
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(save_root, "group2", f"img_{i}.png"))

    return None

if __name__ == "__main__":
    make_imgs_and_labels("/home/nicholas/Datasets/randn")
