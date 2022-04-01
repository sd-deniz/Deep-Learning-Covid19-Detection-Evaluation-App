import torch
import torchvision.transforms as transforms
from PIL import Image


class WrongModelInputSize(Exception):
    pass


class NotATorchSensor(Exception):
    pass


def apply_transforms(img) -> torch.tensor:
    """
    Function takes a Covid X-ray Image that is opened with PIL.Image
    as an argument, applies transforms and
    returns the torch.tensor of the image.

    Arguments : a PIL Image
    Returns   : torch.tensor of transformed PIL Image
    """

    # Normalization parameters for transforms.Normalize() transformation.
    mu = 0.570406436920166
    sigma = 0.1779220998287201

    # PIL library has to be used for convertion.
    # Images that used to train model followed same convertion process
    # which is: img > PIL.convert('L') > PIL.convert('RGB')
    img = img.convert("L")
    img = img.convert("RGB")

    transformation = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((mu,), (sigma,))]
    )

    # apply transformations to the image
    transformed_img = transformation(img)

    return transformed_img


def make_input_for_nn_model(img_path: str) -> torch.tensor:
    """
    Function takes a Covid X-ray Image path as an argument
    applies transforms with "apply_transform" function and
    returns the torch.tensor of the image.

    Argument:
        Test Image path
    Returns:
        Input tensor for a pytorch neural network model
    """
    # open image
    img = Image.open(img_path)

    # apply transforms to covid xray image
    transformed_img = apply_transforms(img)

    # defensive type control
    if not torch.is_tensor(transformed_img):
        raise NotATorchSensor("Transformed image is not a torch tensor.")

    # add dummy dimension to represent batch size for
    # pytorch neural network model
    transformed_img = transformed_img.unsqueeze(0)

    # check the size of the transformed_img for
    # pytorch neural network model input parameters
    if not (transformed_img.shape == torch.Size([1, 3, 224, 224])):
        raise WrongModelInputSize("Model input image size is not valid.")

    return transformed_img


if __name__ == "__main__":
    import sys
    print("Image Transform module is running as main module.")
    print("Exited.")
    sys.exit()
