from omniglot_dataset import getDataset
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Sampler, DataLoader

NUM_TRAIN_CLASSES = 1100
NUM_VAL_CLASSES = 100
NUM_TEST_CLASSES = 423
NUM_SAMPLES_PER_CLASS = 20

class SplitOmniglotSampler(Sampler):
    """Samples task specification keys for an OmniglotDataset."""
    
    def __init__(self, split_idxs, num_way, num_tasks):
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size= self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks

class Split_Omniglot(Dataset):
    def __init__(self):
        super(Split_Omniglot, self).__init__()
        self.dataset = getDataset()
        self.num_class_sample = 5
        self.num_img_per_class = 5
        self._num_support = 5
        self._num_query = 5

    def __len__(self):
        return self.dataset.shape[0]  

    def __getitem__(self, class_idxs):
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        for label, class_idx in enumerate(class_idxs):
            sampled_indices = np.random.choice(
                list(range(self.dataset.shape[1])),  
                size= self._num_support + self._num_query,
                replace=False
            )
            images = self.dataset[class_idx, sampled_indices]

            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        images_support = torch.stack(images_support)  
        labels_support = torch.tensor(labels_support)  
        images_query = torch.stack(images_query)  
        labels_query = torch.tensor(labels_query)  
        
        return images_support, labels_support, images_query, labels_query

    def identity(self, x):
        return x

    def collate_fn(self, batch):
        support_images, support_labels, query_images, query_labels = zip(*batch)
        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.cat(support_labels, dim=0)
        query_images = torch.cat(query_images, dim=0)
        query_labels = torch.cat(query_labels, dim=0)
        return support_images, support_labels, query_images, query_labels

    def get_omniglot_dataloader(
        self,
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch,
        num_workers=2
    ):
        if split == 'train':
            split_idxs = range(NUM_TRAIN_CLASSES)
        elif split == 'val':
            split_idxs = range(NUM_TRAIN_CLASSES, NUM_TRAIN_CLASSES + NUM_VAL_CLASSES)
        elif split == 'test':
            split_idxs = range(NUM_TRAIN_CLASSES + NUM_VAL_CLASSES, NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES)
        else:
            raise ValueError("Invalid split")

        return DataLoader(
            dataset=self,
            batch_size= batch_size,
            sampler=SplitOmniglotSampler(split_idxs, num_way, num_tasks_per_epoch),
            num_workers=num_workers,
            collate_fn=self.identity,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )





if __name__ == '__main__':
    dataset = Split_Omniglot()
    train_loader = dataset.get_omniglot_dataloader(
        split='test',
        batch_size=16,  
        num_way=5,
        num_support=5,
        num_query=5,
        num_tasks_per_epoch= 600 
    )
    
    print(f'Number of tasks in train_loader: {len(train_loader)}')
    
    # Check the last batch of the DataLoader
    for i, tasks in enumerate(train_loader):
        if i == 0:
            print(f'Batch {i} type: {type(tasks)}')
            print(f'Batch {i} length: {len(tasks)}') 

            # Inspect each task
            for j, task in enumerate(tasks):
                print(f'Task {j} type: {type(task)}')
                print(f'Task {j} length: {len(task)}') 

                try:
                    x_spt, y_spt, x_qry, y_qry = task
                    print(f'Batch {i} support set images shape: {x_spt.shape}')
                    print(f'Batch {i} support set labels shape: {y_spt.shape}')
                    print(f'Batch {i} query set images shape: {x_qry.shape}')
                    print(f'Batch {i} query set labels shape: {y_qry.shape}')
                except ValueError as e:
                    print(f'Error unpacking task: {e}')
                break
            break
