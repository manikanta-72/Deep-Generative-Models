import torchvision
from pathlib import Path
import torchvision.transforms as transforms
from models.NADE import NADE
from model_utils.train import InstanceTrainer
from torch import nn
import torch.optim as optim

if __name__ == "__main__":
    datapath = Path(".").absolute().parent / "data"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(root=str(datapath), train=True, download=True, transform=transform)
    model = NADE(input_dim = 784, hidden_dim = 500)
    loss_fn = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters())

    trainer = InstanceTrainer(
                model = model,
                loss_func = loss_fn,
                optimizer = optimizer
            )
    train_loss = trainer.train_one_epoch(train_set)

    print(train_set[0])
