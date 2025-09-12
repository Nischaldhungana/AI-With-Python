import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 6)

y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

plt.plot(x, y1, '--', label='y = 2x + 1')
plt.plot(x, y2, '-.', label='y = 2x + 2')
plt.plot(x, y3, ':', label='y = 2x + 3')

plt.title('Three Lines with Same Slope')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
