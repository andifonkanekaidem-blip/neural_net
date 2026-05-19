# Neural Network from Scratch (C++)

Textbooks weren't enough. So I built this.

This is my first real C++ learning project — a fully functional feedforward neural network implemented from scratch, no ML libraries, no shortcuts. The goal was to expose the gaps in my C++ knowledge by building something that actually works.

---

## How It Works

You define your layer structure as a `std::vector` of layer configs and pass it into the network constructor along with your inputs. From there, a training loop runs repeated forward and backward passes until the model learns the mapping from your inputs to your target outputs.

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (Adaptive Moment Estimation)
- **Activation:** Leaky ReLU

---

## Usage

```cpp
std::vector<NN_Layer> layers = {
    {inputSize, 8},
    {8, outputSize}
    ,inputs
};

NeuralNetwork net(inputs, layers);
std::vector<float> result= net.forward_pass();
net.backpropagate(result,target);
```

---

## Results

Trained to learn a simple input-to-output mapping:

```
Initial Loss: 21.4631   Final Loss: 0
Output: 4 6

Initial Loss: 18.7928   Final Loss: 0
Output: 4 6

Initial Loss: 25.4406   Final Loss: 0
Output: 4 6
```

Consistent convergence across random initializations.

---

## What I Learned

- Forward pass and backpropagation by hand
- Adam optimizer implementation (moment vectors, bias correction)
- He weight initialization for ReLU-based activations
- Why the Leaky ReLU derivative matters — a zero derivative in the negative region kills gradient flow entirely
- C++ STL internals: `std::vector`, `std::map`, copy/move semantics, rule of five
- Placement new and how `emplace_back` works under the hood

---

## Built With

- C++ (no ML frameworks)
- VS Code
