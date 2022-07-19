# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:44:16 2022

@author: ih345
"""
import numpy as np
import matplotlib.pyplot as plt
from LS_packing  import *
from RSA_packing import *

def clustering(particle_radii,n_particles,packing_fraction_goal):
    """
    From a set of files defining particle statistics from Dream3D the preciptates will be
    packed into n clusters with a set density. The placment uses the 

    Parameters
    ----------
    n_clusters : int
        The number of clusters that are generated.
    size : (1,3) Ndarray
        The dimensions of the volume in which the clusters will be placed.
    packing_fraction_goal : float
        The goal packing density to reach within the clusters.
    """
    
    def plot_spheres(positions,radii,size):
        """
        Visualisation of the sphere placment. Can be used for debugging
    
        Parameters
        ----------
       positions : Ndarray
           The starting positions of the spheres as cartesian coordinates.
       radii : Ndarray
           The radii of the spheres
        size : (1,3) array
            size of the domain in which to plot.
        """
        
        def drawSphere(xCenter, yCenter, zCenter, r):
            #draw sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x=np.cos(u)*np.sin(v)
            y=np.sin(u)*np.sin(v)
            z=np.cos(v)
            # shift and scale sphere
            x = r*x + xCenter
            y = r*y + yCenter
            z = r*z + zCenter
            return (x,y,z)
        
        x = [x[0] for x in positions]
        y = [x[1] for x in positions]
        z = [x[2] for x in positions]
        r = radii
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # draw a sphere for each data point
        for (xi,yi,zi,ri) in zip(x,y,z,r):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs, color="r")
        
        # Set limits
        ax.set_xlim([-size, size])
        ax.set_ylim([-size, size])
        ax.set_zlim([-size, size])

        
        plt.show()
    
    # From the density of the clusters find the diameter of the cluster 
    sphere_radius_0 = particle_radii*((n_particles/0.2) ** (1./3)) 
    radii           = np.ones(n_particles)*particle_radii
    
    # place via RSA
    print("Placing spheres via RSA")
    positions = RSA_packing(radii, sphere_radius_0, maxiter=1e4)
    
    print("Placing spheres via LS")
    sphere_radius = max([(pos[0]**2+pos[1]**2+pos[2]**2)**(1./2) for pos in positions])+np.mean(radii)
    positions = LS_packing(positions,radii,sphere_radius,packing_fraction_goal)

    
    sphere_radius = max([(pos[0]**2+pos[1]**2+pos[2]**2)**(1./2) for pos in positions])+np.mean(radii)
    print('Reached a density of '+str(n_particles*(np.mean(radii)/sphere_radius)**3) +' in the cluster')
    
    # Show the spheres in the box, used for bebugging
    plot_spheres(positions,radii,sphere_radius*1.25)
    
    return positions

# Example
particle_radii        = 8
packing_fraction_goal = 0.3
n_particles           = 10

pos = clustering(particle_radii,n_particles,packing_fraction_goal)