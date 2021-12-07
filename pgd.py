import imagenet_stubs
import torch
from tensorflow.keras.preprocessing import image
import numpy as np
from MyVGG2 import MyVGG2


def pgd_attack(model, images, labels, eps=0.1, alpha=1 / 255, iters=10, targeted=True):
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign() * (1 - 2 * int(targeted))
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images.numpy()


if __name__ == '__main__':
    vgg = MyVGG2()

    target_image_name = 'koala.jpg'
    init_image_name = 'tractor.jpg'
    target_image = None
    init_image = None
    for image_path in imagenet_stubs.get_image_paths():
        if image_path.endswith(target_image_name):
            target_image = image.load_img(image_path, target_size=(224, 224))
            image.save_img('target_img.png', target_image)
            target_image = image.img_to_array(target_image) / 255.
            target_image = np.expand_dims(target_image.transpose(2, 0, 1), axis=0)
        if image_path.endswith(init_image_name):
            init_image = image.load_img(image_path, target_size=(224, 224))
            image.save_img('init_img.png', init_image)
            init_image = image.img_to_array(init_image) / 255.
            init_image = np.expand_dims(init_image.transpose(2, 0, 1), axis=0)

    init_label = vgg(torch.from_numpy(init_image))
    init_label = init_label.detach().numpy()

    target_label = vgg(torch.from_numpy(target_image))
    target_label = target_label.detach().numpy()

    #
    if len(init_label.shape) >= 2:
        preds_max = np.amax(init_label, axis=1, keepdims=True)
    else:
        preds_max = np.round(init_label)
    y = init_label == preds_max
    init_label = y.astype(np.float32)

    # untargeted attack
    # adv_img = pgd_attack(vgg, init_image, init_label, targeted=False)
    # targeted attack
    adv_img = pgd_attack(vgg, init_image, target_label, targeted=True)

    image.save_img('adv_img.png', np.squeeze(adv_img).transpose(1, 2, 0))

    perturbation = adv_img - init_image
    image.save_img('pert_img.png', np.squeeze(perturbation).transpose(1, 2, 0))
