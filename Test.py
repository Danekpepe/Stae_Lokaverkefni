import numpy as np

HW = np.array([[1,2,3], [4,5,6], [7,8,9],[10,11,12]])

energy = np.sum(HW, axis=1)

du = np.diff(HW, axis=1)



usum = np.sum(du,axis=1)

x = np.array([1,2,3,4,5,6,7,8])

y = np.array([5,6,7,8,3,2,1,2])

print(x*y)