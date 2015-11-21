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
flock_colors = ['b','g','c','grey', 'darkviolet','gold']
pred_color = 'r'

#Set number of blocks and number of birds in each flock
#Don't set these too high or else perfomance will suffer!
#These need to ints
num_flocks = 6
num_birds_in_flock = 10

#Flocking settings -try changing these and see what happens! 
#ALL OF THESE SHOULD BE FLOATS!
bird_max_vel = 25.0

sight_radius = 10.0
individual_separation = 1.0
cohesion = 1.0
alignment = 1.0
predator_avoidance = 1.0

#Predator settings
#Note: With the way the simulation is set up, the chases tend to most interesting when the predator is a little slower
#than the birds but can see somewhat further
pred_sight_radius = 20.0
pred_max_vel = 20.0
#Set pred_chase to zero to have the predator ignore the birds
pred_chase = 1

#Rendering speeds
interval = 20.0

#Obstacles
#TODO

""" This function will randomly initialize the flocks"""
def random_init_birds(num_flocks, num_birds_in_flock):
	if num_flocks > len(flock_colors):
		raise "Too many flocks!, Add more colors!"
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
		x = (random.random()-0.5)*2*max_vel
		y = (random.random()-0.5)*2*max_vel
	return (x,y)

"""
Generates forces that cause birds to avoid running into the walls
"""
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

"""
Generates forces that tend to bring birds closer together, at least until they get too close for comfort
"""
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

"""
Generates a force that aligns the movement of nearby members of the flock
"""
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

""" Birds don't want to be too close together! The separation force drives them apart if they are closer
than individual_separation apart"""
def separation_force(flocks):
	forces = np.zeros((len(flocks), len(flocks[0]), 2))
	c =500.0

	for i in xrange(len(flocks)):
		for j in xrange(len(flocks[i])):
			for k in xrange(j+1, len(flocks[i])):
				f = np.subtract(flocks[i][k][0], flocks[i][j][0])
				dist = np.linalg.norm(f)
				if dist < individual_separation:
					forces[i][j] = np.add(forces[i][j], np.multiply(-c/dist, f))
					forces[i][k] = np.add(forces[i][k], np.multiply(c/dist, f))

	return forces

"""Birds always want to be in motion: this force will drive them to try to maintain a constant flight speed of p*max_vel"""
def motive_force(flocks):
    forces = np.zeros((len(flocks), len(flocks[0]), 2))
    c = 10.0
    p = 0.75
    for i in xrange(len(flocks)):
    	forces[i] = map(lambda bird: np.multiply(c*p*bird_max_vel/np.linalg.norm(bird[1]), bird[1]), flocks[i])

    return forces

"""Produces the effect of a predator on the motion of the birds and on the motion of the predator itself
The predator will track towards the closest bird in its sight radius
The birds always try to flee the predator"""
def predator_force(flocks, predator):
	forces = np.zeros((len(flocks), len(flocks[0]), 2))
	pred_force = [0,0]
	c = 500.0

	min_distance_bird = [100, None]
	for i in xrange(len(flocks)):
		for j in xrange(len(flocks[i])):
			f = np.subtract(predator[0], flocks[i][j][0])
			dist = np.linalg.norm(f)
			if dist < min_distance_bird[0] and dist < pred_sight_radius:
				min_distance_bird = [dist, -f]
			if dist < sight_radius:
				forces[i][j] = np.add(forces[i][j], np.multiply(-predator_avoidance*c/dist, f))

	if min_distance_bird[1] != None and pred_chase > 0:
		pred_force = np.multiply(10.0,min_distance_bird[1])

	return pred_force,forces

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
patches = [[plt.Polygon(gen_polygon(flocks[i][j][0], flocks[i][j][1]), alpha=1, color=flock_colors[i]) for j in xrange(len(flocks[i]))] for i in xrange(len(flocks))]
#Oh no it's a hawk!
predator = [(95*random.random()+5, 95*random.random()+5), generate_random_velocity_vector(bird_max_vel)]
pred_patch = plt.Polygon(gen_polygon(predator[0], predator[1]), alpha=1, color=pred_color)

#Initialize the birds in the graph
def init():
	return ax

def animate(i):
	npatches = []
	global flocks
	global patches

	#If this is the first frame, initialize the view!
	if i==0:
		#Add all of our birds to the window
		for flock in patches:
			for patch in flock:
				plt.gca().add_patch(patch)
				npatches.append(patch)
		#Add the predator to the window
		plt.gca().add_patch(pred_patch)
		npatches.append(pred_patch)
		return npatches

	#Calculate all the forces acting on the birds
	wall_force = avoid_wall_obstacle_force(flocks)
	coh_force = cohesion_force(flocks)
	align_force = alignment_force(flocks)
	mot_force = motive_force(flocks)
	sep_force = separation_force(flocks)
	pred_force, flee_force = predator_force(flocks, predator)
	#Sum the forces to obtain the total force on each bird!
	total_force = sum_arrays(wall_force, coh_force, align_force, mot_force, sep_force, flee_force)
	for i,flock in enumerate(flocks):
		#Update position and velocities using velocity and force: assume birds have mass 1
		positions = np.add(map(lambda bird: bird[0], flock), np.multiply(interval/1000.0,map(lambda bird: bird[1], flock)))
		velocities = np.add(map(lambda bird: bird[1], flock), np.multiply(interval/1000.0,total_force[i]))
		#Cap velocities at max_vel
		velocities = map(lambda velocity: velocity if np.linalg.norm(velocity) < bird_max_vel else np.divide(velocity, np.linalg.norm(velocity)/bird_max_vel), velocities)
		#Update our flocks with the new positions
		flocks[i] = map(lambda position, velocity: (position, velocity), positions, velocities)
		newpolys = map(lambda bird: gen_polygon(bird[0], bird[1]), flock)
		for j,poly in enumerate(newpolys):
			#patches[i][j].set_alpha(1)
			patches[i][j].set_xy(poly)
			npatches.append(patches[i][j])

	#Update predator position and info
	pred_wall_force = avoid_wall_obstacle_force([[predator]])[0][0]
	pred_patch.set_xy(gen_polygon(predator[0], predator[1]))
	predator[0] = np.add(predator[0], np.multiply(interval/1000.0, predator[1]))
	predator[1] = np.add(predator[1], np.multiply(interval/1000.0, np.add(pred_force, pred_wall_force)))
	#Cap predator velocity also!
	predator[1] = predator[1] if np.linalg.norm(predator[1]) < pred_max_vel else np.divide(predator[1], np.linalg.norm(predator[1])/pred_max_vel)
	#pred_patch.set_alpha(1)
	npatches.append(pred_patch)

	return npatches

def sum_arrays(*arrays):
	if len(arrays) == 1:
		return arrays[0]
	total = arrays[0]
	for array in arrays[1:]:
		total = np.add(total, array)
	return total


#Animate!
anim = animation.FuncAnimation(fig, animate, 
							   init_func=init, 
							   frames=10000, 
							   interval=interval,
							   blit=True)

plt.show()