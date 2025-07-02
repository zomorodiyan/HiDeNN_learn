import numpy as np
import matplotlib.pyplot as plt

# A2 activation function: second-order (x * x)
def A2(x):
    return x * x

# Hand-crafted neural multiplication module
def multiplication_module(F1_vals, F2_vals):
    outputs = []

    for F1, F2 in zip(F1_vals, F2_vals):
        # Hidden layer activations (A2)
        h1 = A2(F1)       # First hidden neuron
        h2 = A2(F2)       # Second hidden neuron
        h3 = A2(F1 + F2)  # Third hidden neuron

        # Output neuron with corrected predefined weights and no bias
        output = -0.5 * h1 - 0.5 * h2 + 0.5 * h3
        outputs.append(output)

    return np.array(outputs)

# Define the input range
x_vals = np.linspace(0, 2 * np.pi, 500)
F1_vals = np.sin(x_vals)
F2_vals = np.cos(x_vals)

# Neural network approximation of the product
nn_product_vals = multiplication_module(F1_vals, F2_vals)

# True product
true_product_vals = F1_vals * F2_vals

# Plot the results
plt.figure(figsize=(10, 8))

# Plot F1
plt.subplot(3, 1, 1)
plt.plot(x_vals, F1_vals, label='$F_1(x) = \sin(x)$', color='blue')
plt.grid(True)
plt.legend()

# Plot F2
plt.subplot(3, 1, 2)
plt.plot(x_vals, F2_vals, label='$F_2(x) = \cos(x)$', color='green')
plt.grid(True)
plt.legend()

# Plot true product and NN approximation
plt.subplot(3, 1, 3)
plt.plot(x_vals, nn_product_vals, label='NN Approximation', color='red')
plt.plot(x_vals, true_product_vals, label='True $F_1 \cdot F_2$', linestyle='--', color='black')
plt.grid(True)
plt.legend()
plt.xlabel('$x$')

plt.suptitle('Neural Network Approximation of $F_1 \cdot F_2$ using $A_2(x) = x^2$')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
