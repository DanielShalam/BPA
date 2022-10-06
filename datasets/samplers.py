import torch
import numpy as np
torch.random.manual_seed(1)


class CategoriesSampler:

    def __init__(self, set_name, labels, num_episodes, num_way, num_shot, num_query):
        self.set_name = set_name
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.num_episodes = num_episodes
        self.m_ind = []
        self.batches = []

        labels = np.array(labels)
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        if self.set_name != 'train':
            for i_batch in range(self.num_episodes):
                batch_gallery = []
                batch_query = []
                classes = torch.randperm(len(self.m_ind))[:self.num_way]
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch_gallery.append(l[pos[: self.num_shot]])
                    batch_query.append(l[pos[self.num_shot: self.num_shot + self.num_query]])

                batch = torch.cat(batch_gallery + batch_query)
                self.batches.append(batch)

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        if self.set_name == 'train':
            for batch_idx in range(self.num_episodes):
                batch_gallery = []
                batch_query = []
                classes = torch.randperm(len(self.m_ind))[:self.num_way]
                for c in classes:
                        l = self.m_ind[c.item()]
                        pos = torch.randperm(l.size()[0])
                        batch_gallery.append(l[pos[: self.num_shot]])
                        batch_query.append(l[pos[self.num_shot: self.num_shot + self.num_query]])

                batch = torch.cat(batch_gallery + batch_query)
                yield batch
        else:
            for batch_idx in range(self.num_episodes):
                # batch = torch.stack(self.batches[i_batch]).reshape(-1)
                yield self.batches[batch_idx]
