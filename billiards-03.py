import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# ======================== Parameters =========================
# Ellipse and attraction field parameters
a, b = 4, 2                          # Ellipse semi-axes
attraction_point = [0.0, 0.0]          # Attraction center (origin)
attraction_radius = 0.5                # Field radius
gravity = 0.000                        # Gravity magnitude (try 0.0 for no gravity)

# Initial state (matching your animation code)
initial_position = [0.5, -1.5]
initial_angle = 2 * math.pi / 3
initial_velocity = [math.cos(initial_angle), math.sin(initial_angle)]

# ======================== Plot Setup =========================
fig, (ax_traj, ax_pos_angle) = plt.subplots(1, 2, figsize=(12, 6))
plot_color = 'm'

# Draw the ellipse and the attraction field.
ellipse_patch = Ellipse([0, 0], 2*a, 2*b, edgecolor='b', facecolor='none')
field_patch = Circle(attraction_point, attraction_radius, edgecolor='r', facecolor='r', alpha=0.3)
ax_traj.add_patch(ellipse_patch)
ax_traj.add_patch(field_patch)
ax_traj.set_xlim(-a-1, a+1)
ax_traj.set_ylim(-b-1, b+1)
ax_traj.set_xlabel("x")
ax_traj.set_ylabel("y")
ax_traj.set_title("Trajectory")

ax_pos_angle.set_xlim(-a-1, a+1)
ax_pos_angle.set_ylim(-math.pi, math.pi)
ax_pos_angle.set_xlabel("x-coordinate")
ax_pos_angle.set_ylabel("Angle (radians)")
ax_pos_angle.set_title("(x, angle) Events")

# ======================== Helper Functions =========================

def safe_roots(coeffs):
    """Trim near-zero leading coefficients and compute roots."""
    coeffs = np.trim_zeros(np.array(coeffs), 'f')
    if coeffs.size == 0:
        return np.array([])
    return np.roots(coeffs)

def is_in_field(x, y):
    """Return True if (x,y) lies inside (or on) the attraction field."""
    return math.hypot(x - attraction_point[0], y - attraction_point[1]) <= attraction_radius

def solve_linear_for_field(x, y, vx, vy, r):
    """
    Solve for t > 0 (using constant velocity) such that
       (x + vx*t)^2 + (y + vy*t)^2 = r^2.
    """
    A = vx**2 + vy**2
    B = 2*(x*vx + y*vy)
    C = x**2 + y**2 - r**2
    disc = B**2 - 4*A*C
    if disc < 0:
        return np.inf
    roots = np.roots([A, B, C])
    positive = [t.real for t in roots if np.isreal(t) and t.real > 1e-12]
    return min(positive) if positive else np.inf

def solve_accelerated_for_field(x, y, vx, vy, ax, ay, r):
    """
    Solve for the smallest positive t satisfying
       [x + vx*t + 0.5*ax*t^2]^2 + [y + vy*t + 0.5*ay*t^2]^2 = r^2.
    (This is a quartic in t which reduces to a quadratic if ax and ay are zero.)
    """
    # If acceleration is nearly zero, fall back to linear.
    if math.isclose(ax, 0.0) and math.isclose(ay, 0.0):
        return solve_linear_for_field(x, y, vx, vy, r)
    # Coefficients:
    A4 = 0.25 * (ax**2 + ay**2)
    A3 = vx*ax + vy*ay
    A2 = vx**2 + vy**2 + x*ax + y*ay  # Note: cross-term comes in linearly.
    A1 = 2*(x*vx + y*vy)
    A0 = x**2 + y**2 - r**2
    coeffs = [A4, A3, A2, A1, A0]
    roots = safe_roots(coeffs)
    positive = [t.real for t in roots if np.isreal(t) and t.real > 1e-12]
    return min(positive) if positive else np.inf

def calculate_ellipse_collision(x, y, vx, vy, a, b):
    """
    Solve for t > 0 satisfying
         ((x+vx*t)/a)^2 + ((y+vy*t)/b)^2 = 1.
    Returns (t, x_coll, y_coll) if a solution exists, or None.
    """
    A = (vx**2)/(a**2) + (vy**2)/(b**2)
    if math.isclose(A, 0.0):
        return None
    B = 2*((x*vx)/(a**2) + (y*vy)/(b**2))
    C = (x**2)/(a**2) + (y**2)/(b**2) - 1
    disc = B**2 - 4*A*C
    if disc < 0:
        return None
    t1 = (-B + math.sqrt(disc))/(2*A)
    t2 = (-B - math.sqrt(disc))/(2*A)
    ts = [t for t in (t1, t2) if t > 1e-12]
    if not ts:
        return None
    t_coll = min(ts)
    return t_coll, x + vx*t_coll, y + vy*t_coll

def reflect_off_ellipse(x, y, vx, vy, a, b):
    """
    Compute the reflection of velocity (vx,vy) off the ellipse at (x,y).
    The outward normal (pointing away from the ellipse) is given by
         n = (2x/a^2, 2y/b^2)  (then normalized).
    Reflection: v_new = v - 2(v·n) n.
    """
    n_x = 2*x/(a**2)
    n_y = 2*y/(b**2)
    n_norm = math.hypot(n_x, n_y)
    if math.isclose(n_norm, 0.0):
        return vx, vy
    n_x /= n_norm
    n_y /= n_norm
    dot = vx*n_x + vy*n_y
    return vx - 2*dot*n_x, vy - 2*dot*n_y

# ======================== Next Event Function ========================

def next_event(x, y, vx, vy):
    """
    Given the state (x,y,vx,vy), compute the time and state at the next event.
    Two candidate events are considered:
      • Ellipse collision (using constant-velocity motion).
      • Crossing the attraction field boundary.
    
    For the field event:
      - If gravity == 0, or if the ball is outside the field, use linear motion.
      - If the ball is inside the field and gravity ≠ 0, assume constant acceleration
        computed at the current state:
             a = -gravity · (x,y)/r.
    
    Returns:
       (t_event, x_new, y_new, vx_new, vy_new, event_type)
    with event_type being "ellipse" or "field".
    """
    # Candidate 1: Ellipse collision time (using linear motion).
    ell_sol = calculate_ellipse_collision(x, y, vx, vy, a, b)
    t_ellipse = ell_sol[0] if ell_sol is not None else np.inf

    # Candidate 2: Field boundary crossing.
    if gravity == 0:
        t_field = solve_linear_for_field(x, y, vx, vy, attraction_radius)
        ax_use, ay_use = 0.0, 0.0
    else:
        if is_in_field(x, y):
            r_val = math.hypot(x - attraction_point[0], y - attraction_point[1])
            if math.isclose(r_val, 0.0):
                ax_use, ay_use = 0.0, 0.0
            else:
                # Acceleration always points toward the attraction point.
                ax_use = (attraction_point[0] - x) / r_val * gravity
                ay_use = (attraction_point[1] - y) / r_val * gravity
            t_field = solve_accelerated_for_field(x, y, vx, vy, ax_use, ay_use, attraction_radius)
        else:
            # Outside the field: no acceleration.
            t_field = solve_linear_for_field(x, y, vx, vy, attraction_radius)
            ax_use, ay_use = 0.0, 0.0

    # Choose the next event (the one with the smaller positive time).
    if t_ellipse <= t_field:
        t_event = t_ellipse
        x_new = x + vx*t_event
        y_new = y + vy*t_event
        vx_new, vy_new = reflect_off_ellipse(x_new, y_new, vx, vy, a, b)
        event_type = "ellipse"
    else:
        t_event = t_field
        # For field crossing, update using (possibly) constant acceleration.
        x_new = x + vx*t_event + 0.5*ax_use*t_event**2
        y_new = y + vy*t_event + 0.5*ay_use*t_event**2
        vx_new = vx + ax_use*t_event
        vy_new = vy + ay_use*t_event
        event_type = "field"
    return t_event, x_new, y_new, vx_new, vy_new, event_type

# ======================== Main Simulation Loop ========================

# Initialize state.
x, y = initial_position
vx, vy = initial_velocity

trajectory = []      # List of (x, y) positions.
pos_angle_data = []  # List of (x, angle) events (angle computed from velocity).

n_events = 100
for _ in range(n_events):
    t_event, x, y, vx, vy, event_type = next_event(x, y, vx, vy)
    # Record the state at the event.
    angle = math.atan2(vy, vx)
    pos_angle_data.append((x, angle))
    trajectory.append((x, y))
    # (Optional) Uncomment to debug event details:
    # print(f"Event: {event_type}, dt={t_event:.5f}, pos=({x:.5f}, {y:.5f}), angle={angle:.5f}")

# ======================== Plot Results ========================

# Trajectory plot.
traj_x, traj_y = zip(*trajectory)
ax_traj.plot(traj_x, traj_y, 'o-', color=plot_color, markersize=3, label="Trajectory")
ax_traj.legend()

# (x, angle) plot.
pa_x, pa_angle = zip(*pos_angle_data)
ax_pos_angle.plot(pa_x, pa_angle, 'o', color=plot_color, markersize=3, label="(x, angle)")
ax_pos_angle.legend()

plt.show()
