import os
import sys
import argparse
import torch
from src import image_transform


def print_output(logits: torch.Tensor, output_func_name: str) -> None:
    """
    Function prints output values for given softmax or
    logsoftmax function name.
    Calculates the LogSoftmax or Softmax output.

    Argument:
        logits  (torch.tensor, requires_grad=True):
                    neural network model inference result.
                    The argument is the return object of
                    "get_logits" function.

        output_func_name (str) : 'logsoftmax', 'softmax'

    Returns: None

    """
    if output_func_name == 'logsoftmax':
        output = torch.nn.LogSoftmax(dim=1)(logits).tolist()[0]
    elif output_func_name == 'softmax':
        output = torch.nn.Softmax(dim=1)(logits).tolist()[0]
    else:
        print("Something went wrong in outputFunction.py")

    print(f"\nnn.{output_func_name} Distribution >>>")
    print("---------------")
    print(f"Normal: {output[0]}",
          f"Non-Covid Pneumonia: {output[1]}",
          f"Covid Pneumonia: {output[2]}", sep='\n')
    print("---------------\n")


def get_logits(img_torch_tensor: torch.tensor,
               covid_prediction_model: torch.jit._script.RecursiveScriptModule
              ) -> torch.tensor:
    """
    Returns the inference output of the neural network model.

    Argument:
        img__torch_tensor    (torch.tensor) : transformed image tensor
        CovidPredictionModel (torch.jit._script.RecursiveScriptModule) :
                                        A Neural network model.

    Returns:
        Logits (torch.tensor)
    """
    # output is nn.CrossEntropyLoss for the mode
    logits = covid_prediction_model(img_torch_tensor)

    # returns output of nn.CrossEntropyLoss criterion
    return logits


def load_model(pytorch_model_path: str) -> torch.jit._script.RecursiveScriptModule:
    """
    Loads the pytorch neural network model

    Arguments :
        model_path  (str) : path to the model file

    Returns:
        python object for model inference

            python object type -> torch.jit._script.RecursiveScriptModule
            This object accepts torch tensor as an input.

            Example: inference_output = model_inference(img_torch_tensor)
    """
    try:
        # loads the trained pytorch neural network model, (for CPU).
        covid_prediction_model = torch.jit.load(pytorch_model_path)
    except:
        e = sys.exc_info()[0]
        print("Model could not be loaded.")
        print("Error", e)
        sys.exit()

    # returns a python object for model inference
    # this python object accepts torch tensor as an input
    return covid_prediction_model


def main(args) -> None:

    # get the working directory path
    working_dir = os.getcwd()

    # path, of the pytorch neural network model
    pytorch_model_path = os.path.join(working_dir, "src/ResultModelscriptmodule_CPU.pt")

    # notifies the user. Loading a model may take time.
    print(f"\nEvaluating file -> {args.img}\n")

    # load the model and get the python object of loaded pytorch model
    covid_prediction_model = load_model(pytorch_model_path)

    # get the transformed image as torch.tensor
    transformed_image = image_transform.make_input_for_nn_model(args.img)

    # get the output of nn.CrossEntropyLoss criterion
    logits = get_logits(transformed_image, covid_prediction_model)

    # applies nn.LogSoftmax and nn.Softmax function to logits.
    # prints the result on terminal
    print_output(logits, "logsoftmax")
    print_output(logits, "softmax")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pass a path of an X-ray image")
    parser.add_argument("-img", type=str, required=True, help="path of the image")
    args = parser.parse_args()

    main(args)
