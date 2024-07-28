import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import errno
import urllib.request
import zipfile

class Omniglot(Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)
        self.classes = {v: k for k, v in self.idx_classes.items()} 

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img_path = os.path.join(self.all_items[index][2], filename)

        img = Image.open(img_path).convert('L')
        target = self.idx_classes[self.all_items[index][1]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for url in tqdm(self.urls):
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(file_processed)
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if f.endswith("png"):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


def getDataset():

    """
    Omniglot is a dataset of over 1623 characters from 50 different alphabets (Lake et al.,
    2015). Each character has 20 hand-written images. The dataset is divided into two parts. The first 963
    classes constitute the meta-training dataset whereas the remaining 660 the meta-testing dataset.

    In this case we use the 28 * 28 Images for easier classifications.
    """

    dataset = Omniglot("data", download=True, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ]))

    temp = {}
    for img, label in tqdm(dataset):
        if label in temp:
            temp[label].append(img)
        else:
            temp[label] = [img]

    dataset_np = [np.stack(temp[label]) for label in sorted(temp.keys())]
    dataset_np = np.array(dataset_np, dtype=np.float32)


    dataset = torch.tensor(dataset_np)
    return dataset
    






def visualizeDataset():
    dataset_train , dataset_test , classes = getDataset()
    def show_image(image, label, class_name):
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {label}\nClass: {class_name}")
        plt.axis('off')
        plt.show()
    for i in range(800, 900):
        img = dataset_train[i][0]  
        label = i
        class_name = classes[label] 
        print(f"Sample {i}: Label - {label}, Class - {class_name}, Image shape - {img.shape}")
        show_image(img, label, class_name)

