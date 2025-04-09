import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from scipy.optimize import root

a, b = 4, 2
attraction_point = [0.0, 0.0]
attraction_radius = 0.5
gravity = 1.0

initial_conditions = [
    ([0.0, 1.5], np.pi / 3),
    ([1.5, 0.5], np.pi / 4),
    ([-1.5, 0.5], np.pi / 6),
    ([0.5, -1.5],  2 * np.pi / 3)
]

fig, (ax_position, ax_position_angle) = plt.subplots(1, 2, figsize=(10, 5))
colors = ['b', 'g', 'r', 'm']

ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')
a_point = Circle([attraction_point[0], attraction_point[1]], attraction_radius, edgecolor='r', fc='r', alpha=0.3)
ax_position.add_patch(ellipse)
ax_position.add_patch(a_point)
ax_position.set_xlim(-a-1, a+ 1)
ax_position.set_ylim(-b-1, b+1)
ax_position.set_xlabel('x')
ax_position.set_ylabel('y')

ax_position_angle.set_xlim(-a - 1, 3 * a + 1)
ax_position_angle.set_ylim(0, np.pi)
ax_position_angle.set_xlabel("x-coordinate")
ax_position_angle.set_ylabel("Angle (radians)")

#input: ball position (x,y), velocity vector(vx, vy), ellipse shape(a,b)
#return: reflected velocity vector
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

#input: position, velocity, ellipse
#return: position colliding with ellipse
def calculate_reflection_position_angle(x, y, vx, vy, a, b):
    A = (vx ** 2 / a ** 2) + (vy ** 2 / b ** 2)
    B = 2 * ((x * vx) / a ** 2 + (y * vy) / b ** 2)
    C = (x ** 2 / a ** 2) + (y ** 2 / b ** 2) - 1
    discriminant = B*B - 4*A*C
    if discriminant < 0:
        # No real solutions => no ellipse collision
        return None, None, None, None, None, None

    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    # Among t1 and t2, pick the smaller that is > 0
    possible_times = [t for t in (t1, t2) if t > 1e-12]
    if not possible_times:
        return None, None, None, None, None, None

    t_collision = min(possible_times)

    x_new = x + vx * t_collision
    y_new = y + vy * t_collision
    
    return x_new, y_new, t_collision,0,0,0


def calculate_touching_position_angle_of_attraction_field(x, y, vx, vy, a, b, m, n, gravity, num_field, num_edge, field):
    # Calculate collision with ellipse
    x_new, y_new, t_collision_to_shape, _, _, _ = calculate_reflection_position_angle(x, y, vx, vy, a, b)
    
    # Calculate collision with attraction field using a quartic equation:
    ax = (x - m) * gravity
    ay = (y - n) * gravity
    bx, cx = vx, x - m
    by, cy = vy, y - n
    
    A = ax**2 + ay**2
    B = 2 * (ax * bx + ay * by)
    C = 2 * ax * cx + vx**2 + 2 * ay * cy + vy**2
    D = 2 * (bx * cx + by * cy)
    E = cx**2 + cy**2 - (attraction_radius**2)
    coeffs = [A, B, C, D, E]
    
    try:
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        positive_times = [t for t in real_roots if t > 0]
    except Exception as e:
        print("Error computing roots:", e)
        positive_times = []
        
    if positive_times:
        t_collision_to_attraction_field = min(positive_times)
    else:
        print("no pos sol")
        t_collision_to_attraction_field = float('inf')
    
    # If no collision is found for either, return current state to avoid infinite updates.
    if t_collision_to_shape is None and t_collision_to_attraction_field == float('inf'):
        print("No collision with either shape or attraction field.")
        return x, y, 0, vx, vy, num_field, num_edge, field

    # Use ellipse collision if it occurs first (or if attraction collision time is infinite)
    if t_collision_to_shape is not None and t_collision_to_shape <= t_collision_to_attraction_field:
        num_edge += 1
        vx, vy = reflect_off_ellipse(x_new, y_new, vx, vy, a, b)
        return x_new, y_new, t_collision_to_shape, vx, vy, num_field, num_edge, field
    else:
        # If attraction collision time is infinite, do not update further.
        if t_collision_to_attraction_field == float('inf'):
            print("No attraction collision detected; returning current state.")
            return x, y, 0, vx, vy, num_field, num_edge, field
        print('v diff:', (x - m) * gravity * t_collision_to_attraction_field)
        vx_new = (x - m) * gravity * t_collision_to_attraction_field + vx
        vy_new = (y - n) * gravity * t_collision_to_attraction_field + vy
        x_new = x + vx_new * t_collision_to_attraction_field
        y_new = y + vy_new * t_collision_to_attraction_field
        new_pos_to_field = abs(x_new**2 + y_new**2 - attraction_radius**2)
        print('field pos cal error:', new_pos_to_field, 'at', num_field, 'time')
        print('field position:', x_new, y_new)
        print('original v:', vx, vy)
        print('field velocity:', vx_new, vy_new)
        num_field += 1
        field = True
        return x_new, y_new, t_collision_to_attraction_field, vx_new, vy_new, num_field, num_edge, field


for idx, (initial_position, initial_angle) in enumerate(initial_conditions):
    position = initial_position
    velocity = [np.cos(initial_angle), np.sin(initial_angle)]
    position_angle_data = []
    position_data = []
    num_edge = 0
    num_field = 0

    n_reflection = 4
    for _ in range(n_reflection):
        x_0, y_0 = position
        vx, vy = velocity
        field = False

        x_new, y_new, t_collision, vx, vy, num_field, num_edge, field = calculate_touching_position_angle_of_attraction_field(
            x_0, y_0, vx, vy, a, b, attraction_point[0], attraction_point[1], gravity, num_field, num_edge, field
        )
        
        angle = np.arctan2(vy, vx)
        if y_new > 0:
           position_angle_data.append((x_new + 2 * a, abs(angle)))
        else:
           position_angle_data.append((x_new, abs(angle)))
           
        position = [x_new, y_new]
        position_data.append((position[0],position[1]))

    print('edge:', num_edge)
    print('field:', num_field)
    if position_angle_data:
        pa_x, pa_y = zip(*position_angle_data)
        # p_x, p_y = zip(*position_data)
        ax_position_angle.plot(pa_x, pa_y, 'o', color=colors[idx], markersize=3, label=f'Initial condition {idx + 1}')
        x_coords = [initial_position[0]]+[pos[0] for pos in position_data]
        y_coords = [initial_position[1]]+[pos[1] for pos in position_data]
        ax_position.plot(x_coords, y_coords, marker = 'o', color=colors[idx], markersize=1, label = f'condition {idx + 1}')

ax_position_angle.legend()
ax_position.legend()

plt.show()
