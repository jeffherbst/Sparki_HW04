import pygame
import math
import numpy as np
from RobotLib.Math import *

class ParticleFilter:
    """
    Implements a particle filter for Monte Carlo Localization.
    """
    def __init__(self,num_particles,alpha,robot,omap):
        """ Creates the particle filter algorithm class.
            Arguments:
                num_particles: number of particles
                alpha: list of four coefficients for motion model
                robot: Robot object
                omap: ObstacleMap object
        """
        self.num_particles = num_particles
        self.alpha = alpha
        self.robot = robot
        self.omap = omap

        # make a matrix of particles
        # each row is a particle (cx,cy)
        self.particles = np.zeros((num_particles,2),dtype='float32')
        
        # weight each particle
        self.particle_weights = np.zeros(num_particles,dtype='float32')
        self.particle_weights[:] = 1./num_particles

        # initialize the particles to random positions around the robot's position
        self.particles[:,0] = self.robot.x + np.random.randn(self.num_particles)*10.
        self.particles[:,1] = self.robot.y + np.random.randn(self.num_particles)*10.

    def draw(self,surf):
        """ Draw particles onto a surface
            Args:
                surf: surface to draw on
        """
        #grey color for circle
        color = (128,128,128)

        #size of circle
        radius = 2
        
        #number of particles
        arraySize = self.num_particles - 1
        
        #loop and draw all 200 particles
        for x in range(0, arraySize):
            #print(x)
            xPosition = self.particles[x,0]
            yPosition = self.particles[x,1]
            position = [xPosition, yPosition]
            pygame.draw.circle(surf, color, position, radius)

    def sample_motion_model(self,time_delta):
        """ Simulates Gaussian noise and returns the move matrix for that noise
            Used to update all of the particles
            Arguments:
                time_delta: time elapsed since last update
            Returns:
                T_motion transform matrix
        """
        #current velocities
        random_lin = np.random.normal(0,1)
        random_ang = np.random.normal(0,1)

        #print(random1, random2)
        lin_vel = self.robot.lin_vel + random_lin
        ang_vel = self.robot.ang_vel + random_ang

        #same as normal robot motion
        if lin_vel == 0: #pure linear
            T_motion = transform(lin_vel * time_delta, 0, 0)
        elif ang_vel == 0: #pure rotational
            T_motion = transform(0, 0, ang_vel * time_delta)
        else: #ICC movement
            R = lin_vel / ang_vel
            T_motion = transform(0, R, 0)
            T_motion = T_motion * transform(0, 0, ang_vel * time_delta)
            T_motion = T_motion * transform(0, -R, 0)

        
        return T_motion

    def generate(self, time_delta):
        """ Update partilces by sampling from teh motion model
            Arguments:
                time_delta: time elapsed since last updat
        """
        #same simulated motion for all particles?, based on gaussian randomeness
        T_motion = self.sample_motion_model(time_delta)
        
        #loop thru all particles
        for x in range(0,self.num_particles - 1):
            #get this particles position
            xPosition = self.particles[x,0]
            yPosition = self.particles[x,1]

            #create transform and move it, use robots direction
            T_particle_map = transform(xPosition, yPosition, self.robot.theta)
            T_particle_map = T_particle_map * T_motion 

            #store particles new position back
            self.particles[x,0] = T_particle_map[0,2]
            self.particles[x,1] = T_particle_map[1,2]


    def update(self):
        """ Update particle weights according to the rangefinder reading. """
        pass

    def sample(self):
        """ Re-sample particles according to their weights. """
        # ensure that weights sum up to one
        self.particle_weights /= np.sum(self.particle_weights)

        # calculate the cumulative sum of the particle weights
        sum_particle_weights = np.cumsum(self.particle_weights)
        
        # select N random numbers in the range [0,1)
        sample = np.random.sample(self.num_particles)

        # find which "bin" each random number falls into using binary search
        inds = np.searchsorted(sum_particle_weights,sample)

        # select the new particles according to the bin indices
        new_particles = self.particles[inds,:]
        
        # save the new set of particles
        self.particles = new_particles

