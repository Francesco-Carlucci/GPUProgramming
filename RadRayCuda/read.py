import matplotlib.pyplot as plt
import numpy as np
from gds import read_gds, generate_cubes
from matplotlib.widgets import Slider

STEPS = 100

xv = []
yv = []
zv = []
ev = []
affected_cubes = []
rays = []

with open("./out.txt", "r") as fin:
    # reading the rays
    nrays = int(fin.readline().strip())
    for i in range(nrays):
        splitline = fin.readline().split(",")
        x1 = float(splitline[0]) 
        y1 = float(splitline[1])
        z1 = float(splitline[2])
        x2 = float(splitline[3])
        y2 = float(splitline[4])
        z2 = float(splitline[5])
        rays.append((x1, x2, y1, y2, z1, z2))
    # reading the energy points
    for line in fin:
        e = []
        splitline = line.strip().split(",")
        if len(splitline) == 1:
            affected_cubes.append(int(splitline[0]))
            continue
        x = float(splitline[0])
        y = float(splitline[1])
        z = float(splitline[2])
        for energy in splitline[3:]:
            e.append(energy)
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

# printing the RAYS
for i,r in enumerate(rays):
    if i == 0:
        lw = 5
    else:
        lw = 2
    ax.plot([r[0], r[1]], [r[2], r[3]], [r[4], r[5]], c="maroon", lw=lw)



# coordinates of points, the same in all instances of time
xva = np.array(xv)
yva = np.array(yv)
zva = np.array(zv)
# divide energy value for each step of time
energy_per_step = []
for i in range(STEPS):
    energy_per_step.append(np.array([float(item[i]) for item in ev]))

# scatter, I do this so that colors are normalized wrt to the maximum, if you start with 0 the
# heatmap is too brigth at the end since the scale is normalized of the very small initial values
scat = ax.scatter(xva, yva, zva, c=energy_per_step[STEPS-1])
scat.set_array(energy_per_step[0])

#function caled by slider
def update_time_step(val):
    scat.set_array(energy_per_step[val-1])

# creating the SLIDER
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) 
slider = Slider(ax_slider, label="Time step", valmin=1, valmax=STEPS, valstep=1, valinit=1, orientation="horizontal")
slider.on_changed(update_time_step)

plt.show()