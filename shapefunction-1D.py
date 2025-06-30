import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def shape_function_NI(x_vals, x_I_minus_1, x_I, x_I_plus_1):
    outputs = []

    for x in x_vals:
        # First ReLU layer outputs
        a1 = relu(-x + x_I)        # top branch
        a2 = relu(x - x_I)         # bottom branch

        # Second ReLU layer outputs
        a3 = relu((-1 / (x_I - x_I_minus_1)) * a1 + 1)
        a4 = relu((-1 / (x_I_plus_1 - x_I)) * a2 + 1)

        # Final linear combination
        N_I = a3 + a4 - 1

        outputs.append(N_I)

    return np.array(outputs)

# Node positions
x_I_minus_1 = 0.0
x_I = 0.6
x_I_plus_1 = 1.0

# Evaluate over a range
x_vals = np.linspace(-0.2, 1.2, 500)
N_vals = shape_function_NI(x_vals, x_I_minus_1, x_I, x_I_plus_1)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x_vals, N_vals, label='$N_I(x)$', color='darkorange')
plt.axvline(x_I_minus_1, color='gray', linestyle='--', label='$x_{I-1}$')
plt.axvline(x_I, color='gray', linestyle='--', label='$x_I$')
plt.axvline(x_I_plus_1, color='gray', linestyle='--', label='$x_{I+1}$')
plt.xlabel('$x$')
plt.ylabel('$N_I(x)$')
plt.title('1D Shape Function Realized by Predefined NN Structure (Not a DNN)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
