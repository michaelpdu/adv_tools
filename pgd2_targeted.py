import imagenet_stubs

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from tensorflow.keras.preprocessing import image
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from MyVGG2 import MyVGG2


if __name__ == '__main__':
    vgg = MyVGG2()
    classifier = PyTorchClassifier(model=vgg,
                                   clip_values=(0., 1.),
                                   loss=CrossEntropyLoss(),
                                   nb_classes=1000,
                                   input_shape=(1, 3, 224, 224))

    pgd = ProjectedGradientDescentPyTorch(estimator=classifier,
                                          eps=0.1,
                                          eps_step=1/255,
                                          max_iter=10,
                                          targeted=True,
                                          batch_size=1,
                                          verbose=True)

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

    target_label = np.zeros((1, 1000), dtype=np.float32)
    target_label[0][0] = 1.
    adv_img = pgd.generate(x=init_image, y=target_label)
    image.save_img('adv_img.png', np.squeeze(adv_img).transpose(1, 2, 0))

    perturbation = adv_img - init_image
    image.save_img('pert_img.png', np.squeeze(perturbation).transpose(1, 2, 0))
