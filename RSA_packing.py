# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:08:46 2022

@author: ih345
"""
import numpy as np
import math

def polar2cart(polars):
    """
    Used to turn polar coordinates into cartesian coordinates
    """
    return [polars[0,0] * math.sin(polars[0,1]) * math.cos(polars[0,2]),
         polars[0,0] * math.sin(polars[0,1]) * math.sin(polars[0,2]),
         polars[0,0] * math.cos(polars[0,1])]

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

def RSA_packing(radii, sphere_radius, maxiter=5e4):
    """
    RSA - Random sphere assignment. This function places spheres of set diameters 
    within a set space. The spheres are placed one by one, if the sphere overlaps 
    with any of the other spheres the placment is attempted again until a maximum
    number of attempts defined by maxiter is reached. 

    Parameters
    ----------
    radii : Ndarray
        The radii of the spheres
    sphere_radius : Float64
        The radii of the sphere in which the particles will be packed
    maxiter : TYPE, optional
        The max number of itterations that will be run by the program. The default is 5e4.

    Returns
    -------
    positions : Ndarray
        The updated positions of the spheres as cartesian coordinates.
    """
    
    # Set the radius of the containing sphere such that the placed spheres don't overlap with it
    sphere_radius = sphere_radius - np.mean(radii)
    
    def placement(radii,maxiter,sphere_radius):
        """
        For a set outer radius attempt to place the spheres
        """
        
        # positions of the particles
        positions = []
        
        for i in range(len(radii)):
            
            # attempt to place the particle
            iter_no = 0
            while iter_no <= maxiter+1:
                
                # copy the positions
                pos_copy = positions.copy()
    
                 # find a random point in the structure and append it
                point = polar2cart(np.random.uniform(size=(1, 3)) * (sphere_radius,2*np.pi,2*np.pi))
                pos_copy.append(point)
                
                # check to see if there have been clashes
                clashes = find_clashes(pos_copy,radii)
                
                # if there are no clashes save this point to positions and go to the next particle
                if sum(clashes) == 0:
                    positions = pos_copy
                    break
                
                # if after maxier itterations there is still overlap then it has failed with the set outer radius
                if iter_no == maxiter:
                    # the sphere is too small, increase it's size 10% and try again
                    failure = 1
                    return positions,failure
                    
                iter_no += 1
        failure = 0      
        return failure,positions
    
    # Try place the spheres, if this fails increase the radius of container and retry. 
    attemps = 0  # The number of times the diameter will be increased before failure
    max_attempts = 3 # max increase attempts before failure
    while attemps <= max_attempts:
        failure,positions = placement(radii,maxiter,sphere_radius)
        if failure:
            attemps += 1
            sphere_radius = sphere_radius*1.10
        else:
            break
    
    # Check that every particle has been placed
    if len(positions) < len(radii):
        raise RuntimeError('Failed at initial placement, Try reducing ppt density in the cluster')
                  
    return positions