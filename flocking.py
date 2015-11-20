import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import random

#Setup figure
fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(0, 100))
plt.tight_layout()

#Colors
#Feel free to change!
flock_colors = ['b','g','c','darkgreen']
pred_color = 'r'

#Set number of blocks and number of birds in each flock
#Don't set these too high or else perfomance will suffer!
#These need to ints
num_flocks = 4
num_birds_in_flock = 15

#Flocking settings -try changing these and see what happens! 
#ALL OF THESE SHOULD BE FLOATS!
sight_radius = 10.0
individual_separation = 1.5

cohesion = 1.0
alignment = 1.0
predator_avoidance = 1.0

#Predator settings!
pred_sight_radius = 20.0
pred_max_vel = 25.0

#Rendering speeds
bird_max_vel = 25.0
interval = 20.0

#Obstacles


#Randomly initialize flocks
def random_init_birds(num_flocks, num_birds_in_flock):
	if num_flocks > 4:
		raise "Too many flocks!"
	flocks = []
	for i in xrange(num_flocks):
		flocks.append([])
		for j in xrange(num_birds_in_flock):
			#Each bird is represented by a pair of vectors [position, velocity]
			flocks[i].append([(95*random.random()+5, 95*random.random()+5), generate_random_velocity_vector(bird_max_vel)])
	return flocks

#Generates a random velocity vector with magnitude at most max_vel
def generate_random_velocity_vector(max_vel):
	x,y = max_vel, max_vel
	while np.linalg.norm([x,y]) > max_vel:
		x = random.random()*max_vel
		y = random.random()*max_vel
	return (x,y)

def avoid_wall_obstacle_force(flocks):
	forces = np.zeros((len(flocks), len(flocks[0]), 2))

	#Inner function to calculate the force on the bird from the walls
	def calculate_wall_force(bird):
		c = 1000.0
		minv = 8
		force = [0,0]
		if bird[0][0] < min(sight_radius, minv):
			force[0] += c/bird[0][0]**2
		if 100 - bird[0][0] < min(sight_radius, minv):
			force[0] -= c/(100-bird[0][0])**2
		if bird[0][1] < min(sight_radius, minv):
			force[1] += c/bird[0][1]**2
		if 100 - bird[0][1] < min(sight_radius, minv):
			force[1] -= c/(100-bird[0][1])**2
		return force

	for i in xrange(len(flocks)):
		forces[i] = map(calculate_wall_force, flocks[i])

	return forces

def cohesion_force(flocks):
	forces = np.zeros((len(flocks), len(flocks[0]), 2))
	c = 1.0

	for i in xrange(len(flocks)):
		counts = [1 for k in xrange(len(flocks[i]))]
		for j in xrange(len(flocks[i])):
			for k in xrange(j+1, len(flocks[i])):
				f = np.subtract(flocks[i][j][0], flocks[i][k][0])
				dist = np.linalg.norm(f)
				if dist > individual_separation and dist < sight_radius:
					forces[i][j] = np.add(forces[i][j], np.multiply(-cohesion*c, f))
					forces[i][k] = np.add(forces[i][k], np.multiply(cohesion*c, f))
					counts[i] += 1
					counts[k] += 1
		forces[i] = np.divide(forces[i],np.transpose([counts, counts]))

	return forces

def alignment_force(flocks):
	forces = np.zeros((len(flocks), len(flocks[0]), 2))
	c = 10.0

	for i in xrange(len(flocks)):
		for j in xrange(len(flocks[i])):
			for k in xrange(j+1, len(flocks[i])):
				dist = np.linalg.norm(np.subtract(flocks[i][j][0], flocks[i][k][0]))
				if dist < sight_radius:
					f = np.subtract(flocks[i][k][1], flocks[i][j][1])
					forces[i][j] = np.add(forces[i][j], np.multiply(alignment*c/dist, f))
					forces[i][k] = np.add(forces[i][k], np.multiply(-alignment*c/dist, f))

	return forces

def separation_force(flocks):
	forces = np.zeros((len(flocks), len(flocks[0]), 2))
	c =100.0

	for i in xrange(len(flocks)):
		for j in xrange(len(flocks[i])):
			for k in xrange(j+1, len(flocks[i])):
				f = np.subtract(flocks[i][k][0], flocks[i][j][0])
				dist = np.linalg.norm(f)
				if dist < individual_separation:
					forces[i][j] = np.add(forces[i][j], np.multiply(-c/dist, f))
					forces[i][k] = np.add(forces[i][k], np.multiply(c/dist, f))

	return forces

def motive_force(flocks):
    forces = np.zeros((len(flocks), len(flocks[0]), 2))
    c = 10.0
    p = 0.8
    for i in xrange(len(flocks)):
    	forces[i] = map(lambda bird: np.multiply(c*p*bird_max_vel/np.linalg.norm(bird[1]), bird[1]), flocks[i])

    return forces

#Scales the size of a bird icon
c = 2.0
#Generates the triangle representing a single bird
def gen_polygon(x, v):
	if x[0] > 100 or x[0] < 0 or x[1] > 100 or x[1] < 0:
		error = "Bird out of bounds at " + str(x[0]) + "," + str(x[1])
		raise error
	vn = np.linalg.norm(v)
	return [[x[0] + (v[0]/vn)*(2*c/3), x[1] + (v[1]/vn)*(2*c/3)],
			[x[0] - (v[0]/vn)*(c/3) + (v[1]/vn)*(c/4),   x[1] - (v[1]/vn)*(2*c/3) - (v[0]/vn)*(c/4)],
			[x[0] - (v[0]/vn)*(c/3) - (v[1]/vn)*(c/4),   x[1] - (v[1]/vn)*(2*c/3) + (v[0]/vn)*(c/4)]]

#Stores birds in the simulation
#Each flock is a row of birds, each bird is a pair of vectors [position, velocity]
flocks = random_init_birds(num_flocks, num_birds_in_flock)
patches = [[plt.Polygon(gen_polygon(flocks[i][j][0], flocks[i][j][1]), alpha=0, color=flock_colors[i]) for j in xrange(len(flocks[i]))] for i in xrange(len(flocks))]
print flocks
#patch = plt.Polygon(gen_polygon(x, v), alpha = 0)

#Initialize the birds in the graph
def init():
	npatches = []
	for flock in patches:
		for patch in flock:
			plt.gca().add_patch(patch)
			npatches.append(patch)
	return npatches

def animate(i):
	npatches = []
	global flocks
	global patches
	#Calculate all the forces acting on the birds
	wall_force = avoid_wall_obstacle_force(flocks)
	coh_force = cohesion_force(flocks)
	align_force = alignment_force(flocks)
	mot_force = motive_force(flocks)
	sep_force = separation_force(flocks)
	print sep_force
	total_force = np.add(np.add(wall_force, mot_force), np.add(coh_force, np.add(sep_force,align_force)))
	for i,flock in enumerate(flocks):
		positions = np.add(map(lambda bird: bird[0], flock), np.multiply(interval/1000.0,map(lambda bird: bird[1], flock)))
		velocities = np.add(map(lambda bird: bird[1], flock), np.multiply(interval/1000.0,total_force[i]))
		#Cap velocities at max_vel
		velocities = map(lambda velocity: velocity if np.linalg.norm(velocity) < bird_max_vel else np.divide(velocity, np.linalg.norm(velocity)/bird_max_vel), velocities)
		flocks[i] = map(lambda position, velocity: (position, velocity), positions, velocities)
		newpolys = map(lambda bird: gen_polygon(bird[0], bird[1]), flock)
		for j,poly in enumerate(newpolys):
			patches[i][j].set_alpha(1)
			patches[i][j].set_xy(poly)
			npatches.append(patches[i][j])
	"""
	patch.set_alpha(1)
	x[0] = 5.0 + 3 * np.sin(np.radians(i))
	x[1] = 5.0 + 3 * np.cos(np.radians(i))
	v = (np.cos(np.radians(i)),-np.sin(np.radians(i)))
	patch.set_xy(gen_polygon((x[0],x[1]), v))
	"""
	return npatches

anim = animation.FuncAnimation(fig, animate, 
							   init_func=init, 
							   frames=10000, 
							   interval=interval,
							   blit=True)

plt.show()