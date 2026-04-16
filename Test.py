import numpy as np

HW = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(HW)

energy = np.sum(HW, axis=0)

du = np.diff(HW, axis=0)

print("du:", du)