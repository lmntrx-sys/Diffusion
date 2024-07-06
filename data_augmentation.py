from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, transforms
import matplotlib.pyplot as plt
import numpy as np

def show_example(img, label):
    print("Label: ", label)
    plt.imshow(img.permute(1, 2, 0))

#!wget https://openaipublic.azureedge.net/clip/data/country211.tgz
#!tar zxvf country211.tgz

IMG_SIZE = 128
BATCH_SIZE = 64

def load_transformed_dataset():
  data_transforms = [
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(),
      transforms.Lambda(lambda x: (x * 2) - 1)
  ]
  data_transform = Compose(data_transforms)
  return datasets.ImageFolder(root="country211", transform=data_transform)

dataset = load_transformed_dataset()

# extract a single image for testing
img, label = dataset[10]
show_example(img, label)

def show_tensor_image(image):
  reverse_transform = transforms.Compose([
      transforms.Lambda(lambda x: (x + 1) / 2),
      transforms.Lambda(lambda x: x.permute(1, 2, 0)),
      transforms.Lambda(lambda x: x * 255.),
      transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
      transforms.ToPILImage()
  ])

  if len(image.shape) == 4:
    image = image[0, :, :, :]
  plt.imshow(reverse_transform(image))

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)