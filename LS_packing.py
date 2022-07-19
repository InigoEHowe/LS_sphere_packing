# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:10:13 2022

@author: ih345
"""
import numpy as np
import math

class Environment():
    """
    Defines the space and particle interactions for the LS algorithm
    """
    def __init__(self, radius, dt):
        """
        Initialise the enviroment
        """
        self.radius = radius
        self.dt = dt
        self.particles = []
 
    def update(self):
        """
        update the enviroment for each timestep
        """
        for p1 in self.particles:
            p1.stateUpdate()
            self.bounce(p1)
            for p2 in self.particles:
                if p1 != p2:
                    self.elasticCollision(p1, p2)
                    
    def particlegrowth(self,grow,growthrate):
        """
        Set the particle growth parameters
        """
        self.growparticles = grow
        self.growthrate    = growthrate
            
 
    def addParticle(self, p):
        """
        Place a particle in the enviroment
        """
        self.particles.append(p)
 
    def bounce(self, p):
        """
        Define how the particles interact with the enviroment boundary
        """
        for p in self.particles:
            p.distance_from_centre(p)
            if p.central_dist > self.radius:
                X_unit = (p.X[0]/np.linalg.norm(p.X[0]))
                shift = X_unit*np.linalg.norm(p.central_dist - self.radius)
                p.addPosition(-shift)
                for i in range(len(p.V)):
                    tmp = -2*(np.dot(p.V[0],X_unit)*X_unit)
                    p.addVelocity(tmp)
 
    def elasticCollision(self, p1, p2):
        """
        Define how the particles act when they collide
        """
        dX = p1.X-p2.X
        dist = np.sqrt(np.sum(dX**2))
        if dist < p1.radius+p2.radius:
            offset = dist-(p1.radius+p2.radius)
            p1.addPosition((-dX/dist)*offset/2)
            p2.addPosition((dX/dist)*offset/2)
            dv1 = -np.inner(p1.V-p2.V,p1.X-p2.X)/np.sum((p1.X-p2.X)**2)*(p1.X-p2.X)
            dv2 = -np.inner(p2.V-p1.V,p2.X-p1.X)/np.sum((p2.X-p1.X)**2)*(p2.X-p1.X)
            p1.addVelocity(dv1)
            p2.addVelocity(dv2)
 

class Particle():
    """
    Store the parameters of each particles
    """
    def __init__(self, env, X, V, radius):
        self.env = env
        self.X = X
        self.V = V
        self.radius = radius
 
    def addVelocity(self, vel):
        self.V += vel
 
    def addPosition(self, pos):
        self.X = self.X + pos
        
    def stateUpdate(self):
        """
        With each timestep ove the particle and grow it. 
        """
        self.X = self.X + self.V*self.env.dt
        if self.env.growparticles == True:
            self.radius += self.radius*self.env.dt*self.env.growthrate
        
    def distance_from_centre(self,p):
        """
        Defines the distance of the particle centre to the centre of the enviroment 
        """
        self.central_dist = (np.sum([i ** 2 for i in p.X]))**(0.5)+p.radius

def overlapping(point1,point2,radius1,radius2):
    """
    For two spheres in space with centres poiont1 and point2 and 
    diameters diameter1 and diameter2 this 
    function will return 1 if the spheres surfaces are closer than 
    the specified gap.
    """
    if math.dist(point1,point2) < (radius1+radius2):
        return 1
    else:
        return 0

def find_clashes(ps,radii):
    """
    For a list of locations and diameters this returns a list containing a count
    of how many other ppts each ppt overlaps with as defined by the overlapping function.
    """
    clashes = []
    for i in range(len(ps)):
        temp_ps = ps.copy()
        temp_radii = np.delete(radii, i)
        del temp_ps[i]
        overlap = [overlapping(ps[i],x,radii[i],temp_radii[idx]) for idx, x in enumerate(temp_ps)]
        clashes = np.append(clashes,sum(overlap))
    return clashes

def LS_packing(positions,radii,sphere_radius,packing_fraction_goal,maxiter=2000):
    """
    Algorithm based on the the algorithm developed by (Lubachevsky and Stillinger, 1990).
    This models the particles as hard billiard balls in a sphere with a hard boundary. The
    packing fraction is increased via growing the particles radii with time. 
    
    Parameters
    ----------
    positions : Ndarray
        The starting positions of the spheres as cartesian coordinates.
    radii : Ndarray
        The radii of the spheres
    sphere_radius : Float64
        The radii of the sphere in which the particles will be packed
    packing_fraction_goal : Float64
        The goal packing fraction of the algorithm, it will run until it has reached
        this value or it has done maxiter steps.
    maxiter : int, optional
        The number of itterations that will be run before the program terminates. 

    Returns
    -------
    new_positions : Ndarray
        The updated positions of the spheres as cartesian coordinates.

    """
    
    # initialize physics simulation
    dt = 0.1 # the time step in each step of the simulation
    growthrate = 0.05 # the increase of the particle radius per second as a proprtion of the current radius
    env = Environment(sphere_radius, dt) # set up the envriment in which the simulation will run
    
    # initialize the spheres
    number_of_particles = len(positions)
    for n in range(number_of_particles):
        radius = radii[n]
        X = positions[n] # starting position
        V = np.random.uniform(-1, 1, (1,3)) # starting velocities
        particle = Particle(env, X, V, radius)
        env.addParticle(particle)
    
    # Set the particles to grow with time
    env.particlegrowth(True,growthrate)

    def update_data():
        """
        Run a timestep and return the resulting position and radii
        """
        env.update()
        positions = [env.particles[i].X[0] for i in range(len(env.particles))]
        radii = [env.particles[i].radius for i in range(len(env.particles))]
        return positions, radii   
    
    # Run time steps to increase the packing fraction
    for t in range(maxiter):
        new_positions,new_radii = update_data()
        
        # The radius of the space is defined by the furthest point from the centre that is within a particle
        space_radius = max([(pos[0]**2+pos[1]**2+pos[2]**2)**(1./2) for pos in new_positions])+np.mean(new_radii)
        
        # Check to see if the goal packing fraction has been reached
        if len(positions)*(np.mean(new_radii)/space_radius)**3 > packing_fraction_goal:
            
            # stop particle growth
            env.particlegrowth(False,growthrate)
            
            # Run time steps to eliminate any overlap of the particles
            for t in range(maxiter):
                # check for overlap
                clashes = find_clashes(new_positions,new_radii)
                if sum(clashes) == 0:
                    # change the positions to rescale the particle size
                    new_positions = [pos*np.mean(radii)/(np.mean(new_radii)) for pos in new_positions]
                    return new_positions
                env.update()
                
                # if preventing overlap is impossible this may be an impossible packing fraction
                if t == maxiter-1:
                    RuntimeError('Reached packing fraction but failed to not overlap, reduce goal packing fraction')
    RuntimeError('Reached packing fraction not reached using LS')