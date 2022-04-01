import unittest
from unittest.mock import patch
import sys
import os
import torch
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.image_transform import make_input_for_nn_model, apply_transforms
from src.image_transform import NotATorchSensor, WrongModelInputSize
from evaluate_xray import get_logits, load_model



# path, to the current working directory
current_dir = os.getcwd()

# path, to the "test_files" directory
test_files_dir = os.path.join(current_dir, "test_files")

# path, to the parent directory of "test_run_unittest" module
parent_dir = os.path.dirname(current_dir)

# path, to the pytorch neural network model file
pytorch_model_path = os.path.join(parent_dir, "src/ResultModelscriptmodule_CPU.pt")

# test file, image name (3x500x500 white image)
test_image_name = "whiteRGB500x500.jpeg"

# test file, transformed image pytorch tensor file name
test_transformed_image_tensor_name = "apply_transformed_img.pt"

# test file, logits pytorch file name
# inference output of the "apply_transformed_img.pt"
test_logits_name = "test_logits.pt"


class TestImageTransfrom(unittest.TestCase):

    def test_pytorch_image_transform_tensor_output(self):
        """ Tests whether pytorch image transforms are applied correctly."""
        # open the image
        img_path = os.path.join(test_files_dir, test_image_name)
        img = Image.open(img_path)

        # apply transform
        transformed_img = apply_transforms(img)

        # load precalculated output tensor
        tensor_path = os.path.join(test_files_dir, test_transformed_image_tensor_name)
        comparison_tensor = torch.load(tensor_path)

        # make comparison
        msg = "image_transform file or reference(test) image has changed"
        comparison_result = torch.allclose(transformed_img, comparison_tensor)
        self.assertEqual(comparison_result, torch.tensor(True), msg)

    def test_pytorch_transformed_image_tensor_size(self):
        """
        Tests tensor shape of the transformed image for forward pass.
        (in other words, to test the input size for the
         pytorch neural network model)
        """
        comparison_tensor = torch.zeros(1, 3, 224, 224).shape
        img_path = os.path.join(test_files_dir, test_image_name)
        transformed_img = make_input_for_nn_model(img_path).shape

        # make comparison
        msg = "Dimensions are wrong"
        self.assertEqual(transformed_img, comparison_tensor, msg)

    @patch("src.image_transform.apply_transforms")
    def test_exception_raise_NotATorchSensor(self, mocked_apply_transforms):
        """ Tests make_input_for_nn_model NotATorchSensor exception."""

        mocked_apply_transforms.return_value = "some_test_string"

        # image path, image is not important, any valid path is acceptable
        # "mock" will change the return value of apply_transforms to a
        # a test string
        image_path = os.path.join(test_files_dir, test_image_name)

        with self.assertRaises(NotATorchSensor):
            make_input_for_nn_model(image_path)

    @patch("src.image_transform.apply_transforms")
    def test_exception_raise_WrongModelInputSize(self, mocked_apply_transforms):
        """ Tests make_input_for_nn_model WrongModelInputSize exception."""

        # tensor will be unsqueezed. shape will be (1, 3, 500, 224)
        mocked_apply_transforms.return_value = torch.zeros(3, 500, 224)

        # image path, image is not important, any valid path is acceptable
        # "mock" will change the return value of apply_transforms to a
        # torch zeros tensor with a shape of (3, 500, 224)
        image_path = os.path.join(test_files_dir, test_image_name)

        with self.assertRaises(WrongModelInputSize):
            make_input_for_nn_model(image_path)


class TestEvaluateXray(unittest.TestCase):

    def test_generic_exceptions_for_load_model_function(self):
        """
        Tests exception raise during installation of
        pytorch neural network model.

        ! If the exception is not thrown, the test succeeds.
        """
        # load the model and get the python object of loaded pytorch model
        _ = load_model(pytorch_model_path)

    def test_pytorch_forwardpass_output(self):
        """ Tests whether pytorch inference output correct."""

        # load trained pytorch neural network model (for CPU)
        covid_prediction_model = torch.jit.load(pytorch_model_path)

        # Test forward pass (inference process)
        output = get_logits(torch.ones(1, 3, 224, 224), covid_prediction_model)

        # compare the test result with expected output
        test_output_path = os.path.join(test_files_dir, test_logits_name)
        test_output = torch.load(test_output_path)
        comparison_result = (torch.allclose(output, test_output) == torch.tensor(True))
        self.assertTrue(comparison_result, "Values do not match.")


if __name__ == "__main__":
    unittest.main()
