import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn as nn
from tqdm import tqdm


#--------------------------------------------------
# Settings
#--------------------------------------------------
data_root = "/home/nicholas/Datasets/randn"
num_epochs = 2
device = "cuda"
#--------------------------------------------------


#--------------------------------------------------
# Don't touch 
#--------------------------------------------------
dataset = datasets.ImageFolder(root=data_root, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=5, shuffle=True) 

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(123008, 16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return nn.Sigmoid()(x)


classifier = Classifier().to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters())

def train_loop(epochs: int):
    for epoch in range(epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            guess: torch.Tensor = classifier(batch[0].to(device))
            loss: torch.Tensor = loss_fn(
                guess, batch[1].reshape((-1, 1)).to(torch.float).to(device)
            )
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                batch_loss = loss.item()
            )

            epoch_loss += loss.item()

        print(f"EPOCH {epoch} LOSS {epoch_loss / len(dataloader)}")

if __name__ == "__main__":
    train_loop(epochs=num_epochs)
#--------------------------------------------------
