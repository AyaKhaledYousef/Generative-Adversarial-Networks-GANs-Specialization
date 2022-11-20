import torch
from torch import nn
# from tqdm.auto import tqdm
# from torchvision import transforms
# # from torchvision.datasets import MNIST # Training dataset
# from torchvision.utils import make_grid
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

m = nn.Linear(20, 30)
n=nn.BatchNorm1d(30)
print(m)
print(n)
