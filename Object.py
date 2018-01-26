import numpy as np
import math
import itertools

class Object:
    def __init__(self,id,mu,codebook_val,theta_range,radi_range):
        self.id = id
        self.absorption = mu
        self.r = radi_range[0] + (radi_range[1]-radi_range[0])*np.random.rand()
        self.theta = theta_range[0] +   (theta_range[1]-theta_range[0])*np.random.rand() #in degrees
        self.theta = (self.theta/180)*math.pi
        self.x = self.r * math.cos(self.theta)
        self.y = self.r * math.sin(self.theta)
        self.sector_label = codebook_val

    '''
    def get_sector(self, r,theta,N_rx,N_theta, Rx_coordinates):
        lst1 = list(itertools.product([0, 1], repeat=int(math.ceil(math.log(3,2))))) #3 refers to 3 concentric circles
        lst2 = list(itertools.product([0, 1], repeat=int(math.ceil(math.log(N_theta,2)))))

        radius = [5,10,15] #Rx_coordinates[:,0] #eg: [5,10,15]
        theta_range = np.linspace(0,360,N_theta+1) #eg: [0,90,180,270,360]
        a=''
        b=''

        #check the sector label w.r.t. radius: eg: 0-5 or 5-10 or 10-15
        if r<radius[0]: #eg: < 5 which means 0-5
            a = map(str, lst1[0])[0] + map(str, lst1[0])[1]
        else: #eg: for cases: 5-10 and 10-15
            for i in range(len(radius)-1):
                if r>=radius[i] and r<radius[i+1]:
                    a = map(str, lst1[i+1])[0] + map(str, lst1[i+1])[1]
                    break

        # check the sector label w.r.t. angle: eg: 0-90 or 90-180 or 180-270 or 270-360
        for i in range(len(theta_range)-1):
            if theta>=theta_range[i] and theta<theta_range[i+1]:
                b = map(str, lst2[i])[0] + map(str, lst2[i])[1]
                break

        sector_label = a+b #joining both gives the codeword

        return sector_label
        '''




