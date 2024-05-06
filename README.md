# Multi Perceptron Layer Unit (MPLU)
a proof of concept for what I call multi perceptron layer approach.

## Introduction

This code is a representation of the Multi Perceptron Layer Unit solving the XOR problem as a proof of concept. The MPLU approach utilizes multiple perceptron weights within the same layer, controlled by a trainable internal network. By adjusting transformation factors and diversity coefficients, the layer's parameter count and diversity are increased. This enables training of multiple perceptrons within a single layer, enhancing model expressiveness.

### Equations:
- **Forward propagation**: \( W[ws, Os, Is] = m[ws, Os, u] • w[ws, u, Is] + C[ws, Os, Is] \)
- **Backward propagation**: Gradient calculation and stochastic gradient descent on mutation factors \( m \), \( w \), and \( c \).

## Improvement Suggestions

- **Initializer**: Implement Xavier normal initialization for stability.
- **Optimizer**: Utilize log-loss with the Adam optimizer for noise reduction during training.
- **Gradient Normalization**: Apply gradient normalization techniques to stabilize weight updates.

## Notes

- Model1 achieved an error of 0.333, while Model2 improved to 0.1151. Further tuning and advancement are necessary.
- The impact of adding more layers on model accuracy varies; experimentation is advised.

## Contribute

If you identify any implementation, mathematical, or spelling errors, please open an issue to address them.

## Getting Started

To use MPLU, follow these steps:

1. Install dependencies by running:
   ```
   pip install numpy
   ```

2. Clone the repository:
   ```
   git clone https://github.com/King-Oni/MPLU.git
   ```

4. Example Usage:

    ```python
    from mplu import MPLU

    # Create MPLU model
    model = MPLU(input_size=2, output_size=1, wpi=5, omega=2)

    # Train model
    for epoch in range(epochs):
        error = 0.0
        for x, y in zip(X, Y):
            prediction = model.forward(x)
            error += MSE(y, prediction)
            grads = MSE_grad(y, prediction)
            model.backward(grads, learning_rate)

        error /= len(X)
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch: {epoch+1}/{epochs}, Error: {error}")

    # Evaluate model
    print("Testing")
    print(model.forward([[0], [0]]))
    print(model.forward([[0], [1]]))
    print(model.forward([[1], [0]]))
    print(model.forward([[1], [1]]))
    ```

5. Run the MPLU script:
   ```
   python MPLU.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
