# CustomNet for MNIST Classification

This repository contains a custom neural network architecture, called CustomNet, designed for the MNIST digit classification task. The network utilizes forward binarization during both the forward pass and weight updates while maintaining real values for computation.

## Network Architecture

The CustomNet architecture consists of four fully connected (linear) layers with batch normalization and dropout applied between each layer. The network is structured as follows:

1. **Input Layer:** Accepts input features with dimensionality specified by `in_features`.
2. **Hidden Layer 1:** Fully connected layer with `num_units` neurons.
3. **Hidden Layer 2:** Fully connected layer with `num_units` neurons.
4. **Hidden Layer 3:** Fully connected layer with `num_units` neurons.
5. **Output Layer:** Fully connected layer with `out_features` neurons.

For the forward pass, the network employs binary weights (`binary_fc1`, `binary_fc2`, `binary_fc3`, `binary_fc4`), while real-valued weights (`fc1`, `fc2`, `fc3`, `fc4`) are used for weight updates.

## Usage

To use CustomNet for MNIST classification, follow these steps:

1. Instantiate an instance of the `CustomNet` class, providing the necessary input parameters (`in_features`, `num_units`, `out_features`, etc.).
2. Train the network using appropriate training data and optimization techniques.
3. Evaluate the trained network using test data to assess classification accuracy.

## Accuracy

The accuracy of the trained CustomNet model on the MNIST test dataset is as follows:

- **Test Accuracy (Top-1) with Binarized Weights:** 65.28%
- **Test Accuracy (Top-5) with Binarized Weights:** 96.44%

## Dependencies

- PyTorch (>= 1.0)
- Python (>= 3.6)

## Example

```python
# Example usage of CustomNet
import torch
from custom_net import CustomNet

# Instantiate CustomNet
model = CustomNet(in_features=784, num_units=256, out_features=10)

# Load MNIST data and train the model...
```

## License
This project is licensed under the MIT License.
