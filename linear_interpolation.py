import numpy as np
import matplotlib.pyplot as plt

# Linear interpolation implemented as a neural-like function
class LinearInterpolator:
    def __init__(self, p4, p5, u4, u5):
        self.p4 = p4
        self.p5 = p5
        self.u4 = u4
        self.u5 = u5

    def forward(self, p):
        # Linear interpolation formula
        w4 = (self.p5 - p) / (self.p5 - self.p4)
        w5 = (p - self.p4) / (self.p5 - self.p4)
        return w4 * self.u4 + w5 * self.u5

# Known input/output points
p4, p5 = 1.0, 3.0
u4, u5 = 2.0, 8.0

# Create interpolator model
model = LinearInterpolator(p4, p5, u4, u5)

# Generate inputs and predictions
p_values = np.linspace(p4, p5, 100)
u_values = np.array([model.forward(p) for p in p_values])

# Plot the result
plt.figure(figsize=(8, 4))
plt.plot(p_values, u_values, label='Interpolated $u(p)$')
plt.scatter([p4, p5], [u4, u5], color='red', label='Known points')
plt.xlabel('p')
plt.ylabel('u(p)')
plt.title('Linear Interpolation as Neural Network')
plt.legend()
plt.grid(True)
plt.show()
