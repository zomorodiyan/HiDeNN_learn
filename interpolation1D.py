import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def shape_interpolation_NIuI(x_vals, x_I_minus_1, x_I, x_I_plus_1, u_I):
    outputs = []

    for x in x_vals:
        # First ReLU layer outputs
        a1 = relu(-x + x_I)        # top branch
        a2 = relu(x - x_I)         # bottom branch

        # Second ReLU layer outputs
        a3 = relu((-1 / (x_I - x_I_minus_1)) * a1 + 1)
        a4 = relu((-1 / (x_I_plus_1 - x_I)) * a2 + 1)

        # Third layer: linear + bias -0.5, no activation (A0 is identity)
        a5 = a3 - 0.5
        a6 = a4 - 0.5

        # Final output: scaled by nodal value u_I
        N_I_uI = u_I * (a5 + a6)

        outputs.append(N_I_uI)

    return np.array(outputs)

# Node positions and nodal value
x_I_minus_1 = 0.0
x_I = 0.6
x_I_plus_1 = 1.0
u_I = 2.0  # nodal value to interpolate

# Evaluate over a range
x_vals = np.linspace(-0.2, 1.2, 500)
N_vals = shape_interpolation_NIuI(x_vals, x_I_minus_1, x_I, x_I_plus_1, u_I)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x_vals, N_vals, label='$N_I(x) u_I$', color='blue')
plt.axvline(x_I_minus_1, color='gray', linestyle='--', label='$x_{I-1}$')
plt.axvline(x_I, color='gray', linestyle='--', label='$x_I$')
plt.axvline(x_I_plus_1, color='gray', linestyle='--', label='$x_{I+1}$')
plt.xlabel('$x$')
plt.ylabel('$N_I(x) u_I$')
plt.title('1D Interpolation Function via Predefined NN')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
