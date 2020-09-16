from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from transform_denseNet import RandomCropsTrain


def train_loader(data_dir, batch_size):
    """Takes in the transform and loads the training dataset from Cityscapes class in torch vision"""
    transform_req = RandomCropsTrain()
    data_required = Cityscapes(data_dir, split='train', mode='fine', target_type='semantic', transform=None,
                               target_transform=None, transforms=transform_req)
    city_data_loader = DataLoader(data_required, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return city_data_loader
