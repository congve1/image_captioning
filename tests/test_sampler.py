import unittest

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from image_captioning.data.build import make_batch_data_sampler


class TestSamplers(unittest.TestCase):
    def test_iteration_based_sampler(self):
        dataset = MNIST(download=True, root='.', transform=ToTensor())
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = make_batch_data_sampler(sampler, 8, 10000)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0
        )
        for images, targets in dataloader:
            self.assertEqual(images.size(0), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
