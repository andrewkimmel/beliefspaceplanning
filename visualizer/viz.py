#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

SIZE = 10. 

path = np.loadtxt('./path/path.txt')*SIZE
particles = np.loadtxt('./path/particles.txt')*SIZE
tree = np.loadtxt('./path/tree.txt')*SIZE

# ------------------------------------------------------

plt.figure(1)

rectangle = plt.Rectangle((-10., -10.), 20, 20, fc='c')
plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((-7.5, -7.5), 15, 17.5, fc='y')
plt.gca().add_patch(rectangle)


plt.plot(particles[:,0],particles[:,1],'.r', markersize=0.1)
plt.plot([tree[:,0], tree[:,2]],[tree[:,1], tree[:,3]],'-b', linewidth=.3)

plt.plot(path[:,0],path[:,1],'-k', linewidth=3)
plt.plot(path[0,0], path[0,1],'*k', markersize=5)
plt.plot(path[-1,0], path[-1,1],'ok', markersize=5)

plt.axis('square')
plt.axis([-SIZE, SIZE, -SIZE, SIZE])
plt.xlabel('x')
plt.ylabel('y')
# plt.grid(b=True)
# plt.title('')
plt.show()
