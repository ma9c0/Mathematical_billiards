import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
a, b = 2, 1
step = 0.1


attraction_point = [0.0,-0.9]
attraction_radius = 1
gravity = 0.01

# Initial position and angle
position = [0.0, 1.0]
angle = np.pi / 4  # Initial direction angle
velocity = [np.cos(angle), np.sin(angle)]  # Initial velocity vector

ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')
a_point = Circle([attraction_point[0], attraction_point[1]], 
            attraction_radius, 
            edgecolor='r', fc='r',alpha=0.3)

fig, (ax, ax_phase) = plt.subplots(1, 2)
ax.set_xlim(-a - 1, a + 1)
ax.set_ylim(-a - 1, a + 1)

ax.add_patch(ellipse)
ax.add_patch(a_point)

ball, = ax.plot([], [], 'ro')
trajectory_line, = ax.plot([], [], 'b-', linewidth=0.5)
trajectory = []

ax_phase.set_xlim(-a, a)  # X-axis limits as per the ellipse's x-axis
ax_phase.set_ylim(0, 1.0)  # Adjust Y-axis for tangent velocity
ax_phase.set_xlabel("x-coordinate (at bounce)")
ax_phase.set_ylabel('tangent velocity (at bounce)')
phase_points, = ax_phase.plot([], [], 'bo', markersize=3)
phase_data = []

def check_collision(x, y, a, b):
    # Check if the point is on or outside the ellipse
    return (x ** 2 / a ** 2 + y ** 2 / b ** 2) >= 1

def is_in_field(x,y):
    return math.sqrt(math.pow(x - attraction_point[0],2) + math.pow(y - attraction_point[1],2)) <= attraction_radius


def reflect_off_ellipse(x, y, vx, vy, a, b):
    # Compute the normal at the point (x, y) on the ellipse
    normal_x = 2 * x / a ** 2
    normal_y = 2 * y / b ** 2
    normal_len = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x /= normal_len
    normal_y /= normal_len

    # Reflect the velocity vector based on the normal
    dot_product = vx * normal_x + vy * normal_y
    vx_reflected = vx - 2 * dot_product * normal_x
    vy_reflected = vy - 2 * dot_product * normal_y

    return vx_reflected, vy_reflected


def calculate_tangent_velocity(x, y, vx, vy, a, b):
    # Tangent vector at the current position on the ellipse
    tangent_x = -b ** 2 * x
    tangent_y = a ** 2 * y
    tangent_len = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
    tangent_x /= tangent_len
    tangent_y /= tangent_len

    # Project the velocity onto the tangent vector to get the tangent velocity
    tangent_velocity = vx * tangent_x + vy * tangent_y

    return np.abs(tangent_velocity)  # Use the absolute value of the tangent velocity

def ball_movements(frame):
    global position, velocity

    # Update the ball position
    x_0, y_0 = position
    vx, vy = velocity

    x_new = x_0 + vx * step
    y_new = y_0 + vy * step

    if (is_in_field(x_new, y_new)):
        down = [attraction_point[0] - x_new, attraction_point[1] - y_new]
        magnitude = math.sqrt(math.pow(down[0],2) + math.pow(down[1],2))
        vx += (down[0]/magnitude) * gravity
        vy += (down[1]/magnitude) * gravity

        velocity = [vx, vy]

        x_new = x_0 + vx * step
        y_new = y_0 + vy * step

    # Check if the ball collides with the ellipse boundary
    if check_collision(x_new, y_new, a, b):
        # Reflect the velocity vector if there is a collision
        vx, vy = reflect_off_ellipse(x_0, y_0, vx, vy, a, b)
        x_new = x_0 + vx * step
        y_new = y_0 + vy * step
        velocity = [vx, vy]

        # Calculate the tangent velocity just after the collision
        tangent_velocity = calculate_tangent_velocity(x_0, y_0, vx, vy, a, b)

        # Append the x-coordinate and tangent velocity to the phase space data
        phase_data.append((x_0, tangent_velocity))
        phase_x, phase_y = zip(*phase_data)
        phase_points.set_data(phase_x, phase_y)

    position = [x_new, y_new]

    # Update the trajectory (draw continuous line on the left plot)
    trajectory.append((x_new, y_new))
    trajectory_x, trajectory_y = zip(*trajectory)
    trajectory_line.set_data(trajectory_x, trajectory_y)
    ball.set_data([x_new], [y_new])  # Update ball position with x and y as separate sequences
    return ball, trajectory_line, phase_points


# Create the animation
ani = FuncAnimation(fig, ball_movements, frames=200, interval=0.1, blit=True)

plt.show()