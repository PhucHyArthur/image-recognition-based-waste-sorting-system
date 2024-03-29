from torchvision import transforms
import torch
from utils.image import remove_background_of_image
# from keras.applications.resnet import preprocess_input
from keras.applications.densenet import preprocess_input
import keras.utils as image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def resize(img_array):
    return img_array.resize((224, 224))


def remove_last_layer_of_transparent_image(img_array):
    return img_array[:-1]


def img_to_array(img):
    return image.img_to_array(img).reshape((3, 224, 224))


def copy(img_array):
    return img_array.copy()


data_transforms = {
    'train':
    transforms.Compose([
        # transforms.Resize((224,224)),
        resize,
        img_to_array,
        # remove_background_of_image,
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # remove_last_layer_of_transparent_image,
        # normalize
        preprocess_input,
        copy,
        torch.from_numpy
    ]),
    'validation':
    transforms.Compose([
        # transforms.Resize((224,224)),
        resize,
        img_to_array,
        # remove_background_of_image,
        # transforms.ToTensor(),
        # remove_last_layer_of_transparent_image
        # normalize
        preprocess_input,
        copy,
        torch.from_numpy
    ]),
    'test':
    transforms.Compose([
        # transforms.Resize((224,224)),
        resize,
        img_to_array,
        # remove_background_of_image,
        # transforms.ToTensor(),
        # remove_last_layer_of_transparent_image
        # normalize
        preprocess_input,
        copy,
        torch.from_numpy
    ]),
}


data_transforms = {
    'train':
    transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'validation':
    transforms.Compose([
        resize,
        transforms.ToTensor()
    ]),
    'test':
    transforms.Compose([
        resize,
        transforms.ToTensor()
    ]),
}