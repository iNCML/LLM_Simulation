import numpy as np
import matplotlib.pyplot as plt

# block_size = 32
obj32 = [4.3976, 1.2540, 1.2019, 1.1449, 1.1263, 1.1048, 1.1125, 1.1085, 1.0967, 1.0834, 1.0812]
# block_size = 64
obj64 = [3.1384, 1.2177, 1.1654, 1.0964, 1.0469, 1.0419, 1.0299, 1.0347, 1.0181, 1.0237, 1.0170]
# block_size = 128
obj128= [3.6253, 1.1647, 1.1416, 1.0547, 0.9969, 1.0020, 0.9911, 0.9942, 1.0014, 0.9886, 0.9870]

fig = plt.figure()
x = np.arange(10)

plt.plot(x, obj32[1:], 'g*-', label='transformer (block=32)')
plt.plot(x, obj64[1:], 'bo-', label='transformer (block=64)')
plt.plot(x, obj128[1:], 'r^-', label='transformer (block=128)')

plt.xlabel('amount of training data')
plt.ylabel('cross-entropy objective function')
plt.xticks([1,3,5,7,9])

plt.legend(loc='middle right')

plt.show()
