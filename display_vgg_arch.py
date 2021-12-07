import torchvision.models as models
from torchsummary import summary
import torch


if __name__ == '__main__':
    vgg16 = models.vgg16()
    stat_dict = torch.load('/Users/dupei/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
    vgg16.load_state_dict(stat_dict)
    summary(vgg16, input_size=(3, 224, 224))
