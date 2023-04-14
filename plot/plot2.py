import matplotlib.pyplot as plt

fig = plt.figure()
x = range(3)

fig, ax = plt.subplots()

epsilon = ["$\epsilon=0.0$", "$\epsilon=0.03$","$\epsilon=0.06$"]
bounds = [0.059, 0.120, 0.139]
bar_labels = ['red', 'blue', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:orange']

ax.bar(epsilon, bounds, label=bar_labels, color=bar_colors)

ax.set_ylabel('average KL divergence')
ax.set_title('language ambiguity')
#ax.legend(title='Fruit color')

plt.show()

