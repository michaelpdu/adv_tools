import torchvision.models as models
from torchsummary import summary
import torch
from torch import nn
from torch.nn.functional import normalize, softmax
from torchvision import transforms
from image_util import *


class MyVGG2(nn.Module):
    def __init__(self):
        super(MyVGG2, self).__init__()
        vgg16 = models.vgg16()
        stat_dict = torch.load('/Users/dupei/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
        vgg16.load_state_dict(stat_dict)
        self.vgg16 = vgg16

        self.preprocess = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.vgg16(x)
        x = softmax(x, dim=1)
        return x


if __name__ == '__main__':
    myvgg = MyVGG2()
    myvgg.eval()

    x = load_image('init_img.png')
    result = myvgg(torch.from_numpy(x))
    label = torch.argmax(result, dim=1)
    print('init_img label:', label, ', and prob:', result[0][label])

    y = load_image('target_img.png')
    result_y = myvgg(torch.from_numpy(y))
    label_y = torch.argmax(result_y, dim=1)
    print('target_img label:', label_y, ', and prob:', result_y[0][label_y])

    x_adv = load_image('adv_img.png')
    result_adv = myvgg(torch.from_numpy(x_adv))
    label_adv = torch.argmax(result_adv, dim=1)
    print('adv_img label:', label_adv, ', and prob:', result_adv[0][label_adv], ', and prob(original):', result[0][label])
