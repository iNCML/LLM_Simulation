import numpy as np
import matplotlib.pyplot as plt

# block_size = 32
diff32 = [3.26183, 0.21254, 0.16093, 0.09788, 0.08263, 0.07394, 0.07734, 0.07409, 0.05981, 0.05658, 0.05146]
# block_size = 64
diff64 = [2.15734, 0.19281, 0.14904, 0.08154, 0.03645, 0.03370, 0.02970, 0.03206, 0.02404, 0.02599, 0.02196]
# block_size = 128
diff128= [2.60530, 0.15512, 0.13383, 0.04841, 0.01885, 0.01775, 0.02241, 0.02107, 0.02006, 0.02300, 0.02201]

fig = plt.figure()
x = np.arange(10)

plt.plot(x, diff32[1:], 'g*-', label='transformer (block=32)')
plt.plot(x, diff64[1:], 'bo-', label='transformer (block=64)')
plt.plot(x, diff128[1:], 'r^-', label='transformer (block=128)')

plt.xlabel('amount of training data')
plt.ylabel('average probability deviation')
plt.xticks([1,3,5,7,9])

plt.legend(loc='middle right')

plt.show()
