from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from transform_PSPNet import *


def test_loader(data_dir, batch_size):
    """Takes in the transform and loads the training dataset from Cityscapes class in torch vision"""
    transform_req = TestTransform()
    data_required = Cityscapes(data_dir, split='val', mode='fine', target_type='semantic', transform=None,
                               target_transform=None, transforms=transform_req)
    city_data_loader = DataLoader(data_required, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return city_data_loader
