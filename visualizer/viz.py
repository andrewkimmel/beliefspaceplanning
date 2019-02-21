#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/toy_simulator/src/')
import varz as V

SIZE = V.SIZE 

path = np.loadtxt('./path/path.txt')*SIZE
particles = np.loadtxt('./path/particles.txt')*SIZE
tree = np.loadtxt('./path/tree.txt')*SIZE

# ------------------------------------------------------

plt.figure(1)

# rectangle = plt.Rectangle((-1, -1), 0.1, 2, fc='w')
# plt.gca().add_patch(rectangle)
# rectangle = plt.Rectangle((0.9, -1), 0.1, 2, fc='w')
# plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((-V.B, -1*V.SIZE), 2*V.B, V.SIZE/2, fc='b')
plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((-V.B, -0.5*V.SIZE), 2*V.B, V.SIZE/2, fc='m')
plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((-V.B, -0.), 2*V.B, V.SIZE/2, fc='g')
plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((-V.B, 0.5*V.SIZE), 2*V.B, V.SIZE/2, fc='c')
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
