import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle

from queue import Queue

from lib.wall import wall


walls = [
    wall([1,2],[1,-2],[1,-1]),
    wall([-1,2],[-1,-2],[-5,-1]),
    #right left
    wall([-1,-2],[1,-2],[1, -2]),
    wall([-1,2],[1,2],[0,0])
    #bottom top
]
# a rectangle

# Ellipse parameters and initial conditions
a, b = 2, 1

e = 0.00001

# Initial position and angle
position = [0.0, 0.0]
angle = e
velocity = [np.cos(angle), np.sin(angle)]

# Setting up the plots

fig, (ax, ax_phase, ax_position_angle) = plt.subplots(1, 3, figsize=(15, 5))

ax.set_xlim(-a - 1, a + 1)
ax.set_ylim(-a - 1, a + 1)

ball, = ax.plot([], [], 'ro')
trajectory_line, = ax.plot([], [], 'b-', linewidth=0.5)
trajectory = []

# Phase space setup
ax_phase.set_xlim(-a, a)
ax_phase.set_ylim(0, 1.0)
ax_phase.set_xlabel("x-coordinate (at bounce)")
ax_phase.set_ylabel("tangent velocity (at bounce)")

phase_points, = ax_phase.plot([], [], 'bo', markersize=3)
phase_data = []

# Position vs. Angle plot setup
ax_position_angle.set_xlim(-a - 1, a + 1)
ax_position_angle.set_ylim(-np.pi, np.pi)
ax_position_angle.set_xlabel("x-coordinate")
ax_position_angle.set_ylabel("Angle (radians)")
position_angle_points, = ax_position_angle.plot([], [], 'go', markersize=1.5)
position_angle_data = []

for w in walls:
    w.createPatch(ax)

def reflect_off_ellipse(x, y, vx, vy, a, b):
    normal_x = 2 * x / a ** 2
    normal_y = 2 * y / b ** 2
    normal_len = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x /= normal_len
    normal_y /= normal_len
    dot_product = vx * normal_x + vy * normal_y
    vx_reflected = vx - 2 * dot_product * normal_x
    vy_reflected = vy - 2 * dot_product * normal_y
    return vx_reflected, vy_reflected

def calculate_tangent_velocity(x, y, vx, vy, a, b):
    tangent_x = -b ** 2 * x
    tangent_y = a ** 2 * y
    tangent_len = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
    tangent_x /= tangent_len
    tangent_y /= tangent_len
    tangent_velocity = vx * tangent_x + vy * tangent_y
    return np.abs(tangent_velocity)

def processWalls(wallList : list) -> tuple:
    if (len(wallList) == 0):
        return None
    
    art = [] * 5
    #run sorting algo based on distance, idc

    for w in wallList:
        intersect, tangentVelocity = w.getIntersect(position, velocity)
        if intersect != None:
            dist = np.sqrt(
                (position[0] - intersect[0])**2 + (position[1] - intersect[1])**2
            )
            art.append([dist, intersect, tangentVelocity])
    
    closest = 1000
    ret = [None, None]
    for pack in art:
        if (pack[0] < closest):
            closest = pack[0]
            ret = [pack[1], pack[2]]
    
    return ret

def plotAll():
    x_new, y_new = position
    vx, vy = velocity
    
    trajectory.append((x_new, y_new))
    ball.set_data([x_new], [y_new])

    angle = np.arctan2(vy, vx)
    position_angle_data.append((x_new, angle))
    


trajectory.append(position)
trajectory_x, trajectory_y = zip(*trajectory)
trajectory_line.set_data(trajectory_x, trajectory_y)

for i in range(0,64):
   
    new_pos, new_vel = processWalls(walls)
    if (new_pos != None):
        trajectory.append(new_pos)
        trajectory.append(position)
        
    angle += np.pi/32
    velocity = [np.cos(angle), np.sin(angle)]



trajectory_x, trajectory_y = zip(*trajectory)
trajectory_line.set_data(trajectory_x, trajectory_y)

#pa_x, pa_y = zip(*position_angle_data)
#position_angle_points.set_data(pa_x, pa_y)

plt.show()
