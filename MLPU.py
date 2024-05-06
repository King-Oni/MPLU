## [docstring]
"""
this code is a representation of the Multi Perceptron Layer Unit solving the XOR proplem as a proof of concept.
the idea of MPLU is instead of using Multi Layers Perceptrons to raise the model size
we use multiple Perceptron weights in the same layer to raise it size
in this example we're taking the approach of making the weights themselves dependent on a trainable internel network
W[ws, Os, Is] = m[ws, Os, u] • w[ws, u, Is] + C[ws, Os, Is] thus raising the weights diversity chance
while also being able to control said diversity thus raisning the amount of parameters the layer have
we alos alwoed for multiple weights to be in one layer thus being able to train multiple perceptrons with in the same layer

W: refers to the weights of the perceptron
while m, w are considered a transformation factors
c: is a way to raise the wights value either up or down

ws: refers to the number of perceptrons with in the layer
while Os is the output size, and Is is the input size respectively

u: on the other hand refers to the diversity coefficient which can be thought of as a way to give the weights more density

in the forward referance the network first calculate the weights of the perceptrons using the first equation
then it preforms a normal perceptron forward propagation

then in the backward referance the network calculate the gradienta of the mutation factors and raise factor m, w and c
then train them using normal stochastic gradient descent
"""

## [imports]
import numpy as np


## [dataset]
np.random.seed(1337)

# [A, B]
X = [
    [[0], [0]],
    [[1], [0]],
    [[0], [1]],
    [[1], [1]],
]
# Y = X[0] xor X[1]
# [1, 0]:[on, off]
Y = [
    [[0]],
    [[1]],
    [[1]],
    [[0]],
]


## [Perceptron Layer]
class MPLU:
    def __init__(self, input_size: int, output_size: int, wpi: int, omega: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.wpi = wpi
        self.m_shape = (wpi, output_size, omega)
        self.w_shape = (wpi, omega, input_size)
        self.c_shape = (wpi, output_size, input_size)

        self.m = np.random.randn(*self.m_shape)
        self.w = np.random.randn(*self.w_shape)
        self.c = np.random.randn(*self.c_shape)
        self.biases = np.random.randn(wpi, output_size, 1)
        self.tanh = Tanh()

    def __make_weights(self):
        self.weights = np.einsum("wou,wui,woi->woi", self.m, self.w, self.c)

    def forward(self, input):
        self.input = np.array(input)
        self.__make_weights()
        output = np.einsum("wij,jk,wik->ik", self.weights, input, self.biases)
        self.output = output
        return output

    def backward(self, output_gradients, learning_rate=0.0001):
        weights_grad = np.dot(output_gradients, self.input.T)  # oi
        biases_grad = output_gradients
        input_grad = np.einsum("wji,jk->ik", self.weights, output_gradients)

        c_grad = np.zeros(self.c_shape)  # woi
        w_grad = np.einsum("wou,oi->wui", self.m, weights_grad)  # wou
        m_grad = np.einsum("wui,oi->wou", self.w, weights_grad)  # wui
        c_grad = weights_grad

        # learn
        self.m -= m_grad * learning_rate
        self.w -= w_grad * learning_rate
        self.c -= c_grad * learning_rate
        self.biases -= biases_grad * learning_rate

        return input_grad


## [activation]
class Activation:
    def __init__(self, act, grad):
        self.act = act
        self.grad = grad

    def forward(self, input):
        self.input = input
        return self.act(input)

    def backward(self, og, _):
        return np.multiply(og, self.grad(self.input))


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)


## [training]
def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def MSE_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


## [utility functions]
def predict(arch, input):
    for l in arch:
        input = l.forward(input)

    return input


def learn(arch, output_grad, learning_rate):
    for l in reversed(arch):
        output_grad = l.backward(output_grad, learning_rate)

    return output_grad


## [training]
epochs = 100_000
learning_rate = 0.00001
model1 = [MPLU(2, 1, 5, 2)]
model2 = [MPLU(2, 3, 5, 1), Tanh(), MPLU(3, 1, 5, 1)]

for epoch in range(epochs):
    error1 = 0.0
    error2 = 0.0
    for x, y in zip(X, Y):
        prediction1 = predict(model1, x)
        prediction2 = predict(model2, x)
        error1 += MSE(y, prediction1)
        error2 += MSE(y, prediction2)

        grads1 = MSE_grad(y, prediction1)
        grads2 = MSE_grad(y, prediction2)

        learn(model1, grads1, learning_rate)
        learn(model2, grads2, learning_rate)

    error1 /= len(X)
    error2 /= len(X)
    if (epoch + 1) % 1000 == 0:
        print(
            "=============================================#"
            + f"\nepoch: {epoch+1}/{epochs}"
            + f"\nModel1 Error: {error1}"
            + f"\nModel2 Error: {error2}"
            + "\n=============================================#"
        )


## [testing]
print("Testing Model1")
print(predict(model1, [[0], [0]]))
print(predict(model1, [[0], [1]]))
print(predict(model1, [[1], [0]]))
print(predict(model1, [[1], [1]]))

print("Testing Model2")
print(predict(model2, [[0], [0]]))
print(predict(model2, [[0], [1]]))
print(predict(model2, [[1], [0]]))
print(predict(model2, [[1], [1]]))


## [imporovments suggestions]
message = """
the network when training get's sensetive to change in learning rate and are susceptive to change teakable parameters
thus needs a proper initializer.
- better initializer:
    I sugges in this case we use the Xavier normal.

- better optimizer:
    trainging gets noisy some times (most of the time) and I suggest for this problem we use the log-loss with the Adam optimizer.

- gradients normalizatio:
    I suggest also some gradient normalization for the weights.

NOTES:
    - the best model1 ever got was an error of: 0.333 which indicates that it needs more tweaking and and advancing
      while model2 got it up to 0.1151
    - it is not known for sure weather adding more layers makes the model more accurate or not
      sometimes model1 gets faster than model 2 and sometimes else it stumps behind so take this with as much salt as you want
"""

#
