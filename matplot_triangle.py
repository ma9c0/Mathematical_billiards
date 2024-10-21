import matplotlib.pyplot as plt
import numpy as np

import pymunk
from pymunk import Vec2d

from matplotlib.patches import Polygon

from matplotlib.animation import FuncAnimation

fig, (view1, view2) = plt.subplots(1, 2)

view1.set_xlim(-0.1, 1.1)
view1.set_ylim(-0.1, 1.1)

view2.set_ylim(0, 360) 
view2.set_xlim(0,200)

triangle_vertices = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
ball_radius = 0.05
ball_mass = 1
ball_velocity = Vec2d(2, 2)

ball, = view1.plot([], [], 'bo', markersize=10)
trajectory_line, = view1.plot([],[], 'b-', alpha=0.5)

view1.set_xlim(-0.1, 1.1)   
view1.set_ylim(-0.1, 1.1)
view2.set_xlabel('Frame')
view2.set_ylabel('Angle (degrees)')

angleline, = view2.plot([],[], 'bo', markersize=5)

collision_data = []
trail_positions = []

triangle_patch = Polygon(triangle_vertices, closed=True, fill=False, color='black', edgecolor='black')
view1.add_patch(triangle_patch)

def setup():
    global space, ball_body, triangle_vertices
    space = pymunk.Space()
    space.gravity = (0, 0)

    triangle_segments = [
    pymunk.Segment(space.static_body, triangle_vertices[i], triangle_vertices[(i + 1) % 3], 0)
    for i in range(3)
    ]
    for segment in triangle_segments:
        segment.elasticity = 1.0
        space.add(segment)

    

    ball_inertia = pymunk.moment_for_circle(ball_mass, 0, ball_radius)
    ball_body = pymunk.Body(ball_mass, ball_inertia)
    ball_body.position = 0.5, 0.25

    ball_body.velocity = ball_velocity
    ball_shape = pymunk.Circle(ball_body, ball_radius)
    ball_shape.elasticity = 1.0
    space.add(ball_body, ball_shape)

setup()

def calculate_angle(velocity):
    angle = np.degrees(np.arctan2(velocity[1], velocity[0]))
    return angle % 360 

def collision_handler(arbiter, space, data):
    frame = data['frame']
    
    ball_position = ball_body.position
    ball_velocity = ball_body.velocity

    collision_data.append({
        'frame': frame,
        'ball_position': ball_position,
        'ball_velocity': ball_velocity,
        'angle': calculate_angle(ball_velocity)
    })

    return True  


handler = space.add_default_collision_handler()
handler.post_solve = collision_handler

def update(frame):
    handler.data['frame'] = frame

    space.step(0.01)

    ball_position = ball_body.position

    #view1.clear()
    #view2.clear()
    #plot_triangle(view1)
    
    trail_positions.append((ball_position.x, ball_position.y))
    trail_x, trail_y = zip(*trail_positions)
    

    if collision_data:
        frames = [c['frame'] for c in collision_data]
        angles = [c['angle'] for c in collision_data]
        #view2.plot(frames, angles, 'g-')
        angleline.set_data(frames, angles)
        view1.set_title(f'Frame: {frame}, Collisions: {len(collision_data)}')

    #view1.plot(ball_position.x, ball_position.y, 'bo', markersize=10)
    #view1.plot(trail_x, trail_y, 'b-', alpha=0.5)

    ball.set_data([ball_position.x], [ball_position.y])
    trajectory_line.set_data(trail_x,trail_y)

    return ball, trajectory_line, angleline

ani = FuncAnimation(fig, update, frames=200, interval=10, blit=True)

plt.show()
