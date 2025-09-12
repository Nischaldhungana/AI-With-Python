import numpy as np

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

A_inv = np.linalg.inv(A)

I1 = np.dot(A, A_inv)
I2 = np.dot(A_inv, A)

print("A * A_inv =")
print(I1)

print("\nA_inv * A =")
print(I2)









