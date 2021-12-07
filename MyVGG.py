import torchvision.models as models
from torchsummary import summary
import torch
from torch import nn
from torch.nn.functional import normalize
from torchvision import transforms
from image_util import load_image


def loss_func(x, y):
    x = normalize(x)
    y = normalize(y)
    return torch.cdist(x, y)


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        vgg16 = models.vgg16()
        stat_dict = torch.load('/Users/dupei/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
        vgg16.load_state_dict(stat_dict)
        self.features = vgg16.features

        self.preprocess = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.features(x)
        x = x.mean(-1).mean(-1)
        return x


if __name__ == '__main__':
    myvgg = MyVGG()
    myvgg.eval()

    x = load_image('init_img.png')
    result = myvgg(torch.from_numpy(x))
    print(result)
