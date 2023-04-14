import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
x = np.arange(1,5)

x_eps0 = [0.02346811,  0.02404568, 0.02116136, 0.02196647]
x_eps1 = [0.21868669, 0.08282693,0.07681387, 0.04928338 ]
x_eps2 = [0.63307643, 0.18353627, 0.11984389, 0.09501252]

plt.plot(x, x_eps0, 'r*-', label='unambiguous language ($\epsilon=0$)')
plt.plot(x, x_eps1, 'bo-', label='ambiguous language ($\epsilon=0.065$)')
plt.plot(x, x_eps2, 'g^-', label='ambiguous language ($\epsilon=0.157$)')

plt.xlabel('number of prompts')
plt.xticks([1,2,3,4])

plt.legend(loc='middle right')

plt.show()
