import torch
import numpy as np


class CategoriesSampler:

    def __init__(self, set_name, labels, num_episodes,
                 num_way, num_shot, num_query, const_loader, replace=True):

        self.set_name = set_name
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.num_episodes = num_episodes
        self.const_loader = const_loader   # same tasks in different epochs. good for validation
        self.replace = replace             # sample few-shot tasks with replacement (same class can appear twice or more

        self.m_ind = []
        self.batches = []

        labels = np.array(labels)
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.classes = np.arange(len(self.m_ind))

        if self.const_loader:
            for i_batch in range(self.num_episodes):
                batch = []
                # -- faster loading with np.choice -- #
                # classes = torch.randperm(len(self.m_ind))[:self.num_way]
                classes = np.random.choice(self.classes, size=self.num_way, replace=self.replace)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(np.arange(l.shape[0]),
                                           size=self.num_shot + self.num_query,
                                           replace=False)
                    batch.append(l[pos])

                batch = torch.from_numpy(np.stack(batch)).t().reshape(-1)
                self.batches.append(batch)

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        if not self.const_loader:
            for batch_idx in range(self.num_episodes):
                batch = []
                # classes = torch.randperm(len(self.m_ind))[:self.num_way]
                classes = np.random.choice(self.classes, size=self.num_way, replace=self.replace)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(np.arange(l.shape[0]),
                                           size=self.num_shot + self.num_query,
                                           replace=False)
                    batch.append(l[pos])

                batch = torch.from_numpy(np.stack(batch)).t().reshape(-1)
                yield batch
        else:
            for batch_idx in range(self.num_episodes):
                yield self.batches[batch_idx]
