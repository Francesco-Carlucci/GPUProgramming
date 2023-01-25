import matplotlib.pyplot as plt
import numpy as np
from gds import read_gds, generate_cubes

xv = []
yv = []
zv = []
ev = []
affected_cubes = []

with open("./out.txt", "r") as fin:
    splitline = fin.readline().split(",")
    x1 = float(splitline[0]) 
    y1 = float(splitline[1])
    z1 = float(splitline[2])
    x2 = float(splitline[3])
    y2 = float(splitline[4])
    z2 = float(splitline[5])
    for line in fin:
        splitline = line.strip().split(",")
        if len(splitline) == 1:
            affected_cubes.append(int(splitline[0]))
            continue
        x = float(splitline[0])
        y = float(splitline[1])
        z = float(splitline[2])
        e = float(splitline[3])
        xv.append(x)
        yv.append(y)
        zv.append(z)
        ev.append(e)
        
line_x = np.linspace(x1, x2, 1000)
line_y = np.linspace(y1, y2, 1000)
line_z = np.linspace(z1, z2, 1000)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

layers_cube, layer_list, vertex_cube_list = read_gds("../RadrayPy/MUX2_X1_def_45nm.txt")
generate_cubes(vertex_cube_list, affected_cubes, ax)
ax.plot([x1, x2], [y1, y2], [z1, z2], c="maroon", lw=5)
ax.scatter(np.array(xv), np.array(yv), np.array(zv), c=np.array(ev))
  
plt.show()
    