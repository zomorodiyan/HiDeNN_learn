import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def L_dnn_stepwise_debug(x, xA, xB, yA, yB):
    # Step-by-step neuron computation based on diagram
    W1 = -1
    b1 = xB
    a1 = relu(W1 * x + b1)  # First neuron output

    W2 = -1 / (xB - xA)
    b2 = 1
    a2 = relu(W2 * a1 + b2)  # Second neuron output

    W3 = yB - yA
    b3 = yA
    output = W3 * a2 + b3  # Final output

    return a1, a2, output

# Parameters
xA, xB = 0.3, 0.8
yA, yB = 1.0, 3.0

# Domain for plotting
x_vals = np.linspace(0.0, 1.1, 500)

# Compute stepwise outputs
a1_vals, a2_vals, y_final = L_dnn_stepwise_debug(x_vals, xA, xB, yA, yB)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axs[0].plot(x_vals, a1_vals, label='ReLU(-x + x_B)', color='purple')
axs[0].set_ylabel('a1')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(x_vals, a2_vals, label='ReLU((-1 / (x_B - x_A)) * a1 + 1)', color='green')
axs[1].set_ylabel('a2')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(x_vals, y_final, label='(y_B - y_A) * a2 + y_A', color='blue')
axs[2].axvline(xA, color='gray', linestyle='--', label='x_A')
axs[2].axvline(xB, color='gray', linestyle='--', label='x_B')
axs[2].set_ylabel('L(x)')
axs[2].set_xlabel('x')
axs[2].legend()
axs[2].grid(True)

plt.suptitle('L(x) = (y_B - y_A) * ReLU((-1 / (x_B - x_A)) * ReLU(-x + x_B) + 1) + y_A')
plt.tight_layout()
plt.show()
