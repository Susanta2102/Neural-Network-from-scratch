# Neural Network from Scratch

This project demonstrates the implementation of a simple neural network from scratch using Python and NumPy. The goal is to understand the core concepts of neural networks, including weight initialization, forward and backward propagation, gradient descent, and data visualization.

## Overview

The project covers the following key components:

- **Initialization of Weights and Biases**: Setting up the initial parameters for the neural network layers.
- **Activation Functions**: Implementation of ReLU and linear activation functions, along with their derivatives.
- **Forward Propagation**: Calculation of activations as the input data moves through the network layers.
- **Backpropagation**: Computation of gradients to update the network's weights and biases.
- **Training Loop**: Iterative training using gradient descent to minimize the error.
- **Data Visualization**: Visualizing the transformation of input data as it passes through the network.

## Implementation Details

### 1. Initializing Weights and Biases

The function `initialise` sets up the initial weights and biases for the network:
- **`w1` and `w2`**: Weight matrices for the connections between layers.
- **`b1` and `b2`**: Bias vectors for the hidden and output layers.

### 2. Activation Functions

Two activation functions are implemented:
- **ReLU (Rectified Linear Unit)**: Used in the hidden layer to introduce non-linearity.
- **Linear**: Used in the output layer to maintain the continuous nature of the output.

### 3. Forward Propagation

The function `forwardProp` calculates the output of the network:
- **Input Layer -> Hidden Layer**: Computes the activations using the ReLU function.
- **Hidden Layer -> Output Layer**: Computes the final output using the linear function.

### 4. Backpropagation

The function `computeGradient` calculates the gradients of the cost function with respect to weights and biases:
- Gradients are used to update the weights and biases during training to reduce the error.

### 5. Training Loop

The neural network is trained using a loop that:
- Performs forward propagation to get the current output.
- Computes the error and gradients using backpropagation.
- Updates the weights and biases using the computed gradients.

### 6. Data Visualization

Matplotlib is used to visualize:
- **Input Data**: The original input data points.
- **Transformed Data**: The output of the neural network after training, showing how the network maps the input to a new representation.

## Running the Project

1. **Dependencies**:
   - Python 3.x
   - NumPy
   - Matplotlib
   - Pandas (for loading test data)

2. **Running the Code**:
   - Clone the repository:
     ```bash
     git clone https://github.com/Susanta2102/Neural-Network-from-scratch.git
     ```
   - Navigate to the project directory:
     ```bash
     cd Neural-Network-from-scratch
     ```
   - Run the Python script:
     ```bash
     python neural_network_from_scratch.py
     ```

3. **Visualizing the Results**:
   - The final output will include scatter plots showing the transformation of input data by the neural network.

## Example Plots

![Initial Input Data](https://raw.githubusercontent.com/Susanta2102/Neural-Network-from-scratch/main/pic%201.jpeg)
![Transformed Data](https://raw.githubusercontent.com/Susanta2102/Neural-Network-from-scratch/main/pic%204.jpeg)

## Conclusion

This project provides a step-by-step guide to understanding the mechanics of neural networks. By building a network from scratch, you gain insights into how each component works together to learn from data and make predictions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
