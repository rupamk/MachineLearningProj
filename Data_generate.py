''' Class Project Machine Learning: 5523
Authors: Rupam Kundu and Tanmoy Das
Project: Object detection using WiFi'''
#*******Data_generate********

from __future__ import division
from transceiver import transceiver
import numpy as np
import math
import cmath
from random import randint
import itertools
from Object import Object
from decimal import *
from ProgressBar import ProgressBar

class Data_generate():
    def __init__(self,name):
        self.name = name


    def check_sector(self, r, theta):
        pos=0
        t =0
        if r <5:
            pos =1
        elif r>=5 and r<10:
            pos =2
        elif r>=10:
            pos = 3

        if theta<90:
            t=1
        elif theta>=90 and theta<180:
            t=2
        elif theta>=180 and theta<270:
            t=3
        elif theta>=270 and theta<360:
            t=4

        return (pos-1)*4 + t

    def generate_data(self,param):
        '''Set Transmitter's Position'''
        tx_pos = transceiver(param.Tx_Coord)

        # Set Receiver's Position
        Rx = dict()
        N_rx = len(param.Rx_Coord)  # number of receivers
        for i in range(N_rx):
            Rx[i] = transceiver(param.Rx_Coord[i])

        # generate the codebook
        codebook = self.gen_codebook(param.N_radi, param.N_theta)
        #print(codebook)
        codebook_theta = self.gen_codebook_theta(param.N_theta)
        codebook_radius = self.gen_codebook_radius(param.N_radi)
                # for storing all the obj_coordinates:
        obj_coordinate = []

        # all possible combinations of labels
        comb = list(itertools.product([0, 1], repeat=param.N_sector))[1:]
	#print(comb)
        # Initialize Data and labels
        Data = np.zeros([param.N_repeat * len(comb), 2 * param.N_rx])
        labels = np.zeros([param.N_repeat * len(comb), param.N_sector])

        #print the progress bar
        progress = ProgressBar(len(comb), fmt=ProgressBar.FULL)
        count = 0
        a = np.zeros(12)

		
        for i in range(len(comb)):
            val = np.array(comb[i])
            index = np.where(val == 1)[0:]
            #print index
            #print "Len: " + str(len(index[0]))
            progress.current += 1
            progress()
            
            for iter1 in range(param.N_repeat):
                sum = np.zeros(param.N_rx, dtype=complex)

                for iter2 in range(len(index[0])):
                    codebook_val = codebook[index[0][iter2]]

                    #print 'Index: ' + str(index[0][iter2])

                    # first randomly choose "m" out of 10 objects and distribute them in different sectors
                    m = int(math.ceil(param.N_obj * np.random.rand()))  # generate a random number between 1 -10 for 10 objects

                    # for each object compute the received response at each receiver and sum them:
                    obj = dict()

                    for j in range(1,m+1):
                        # upper bound for radius while defining (r,theta) is assumed as the maximum radius at which the Rx is placed : max(max(Coordinates))
                        obj[j] = Object(j, param.Absorption[randint(0, len(param.Absorption) - 1)], codebook_val,
                                        codebook_theta[codebook_val[int(len(codebook_val) / 2):]], codebook_radius[
                                            codebook_val[:int(len(codebook_val) / 2)]])  # Object(id,Absorption,upperbound_radius)

                        # update the global object coordinate store:
                        obj_coordinate.append([obj[j].x, obj[j].y])

                        ind_val = self.check_sector(obj[j].r, math.degrees(obj[j].theta))
                        if index[0][iter2] != (ind_val-1):
                            print(codebook_val)
                            print(codebook_theta[codebook_val[int(len(codebook_val) / 2):]])
                            print(codebook_radius[codebook_val[:int(len(codebook_val) / 2)]])
                            print("Sector does not match, Expected: "+ str(index[0][iter2] + 1) + " Got: "+ str(ind_val) )



                        a[ind_val-1] += 1

                        #a= list(str(i) for i in obj[j].sector_label)
                        #label_val.append([int(i) for i in a])

                        for k in range(param.N_rx):  # for each receiver
                            # compute the distance of the object w.r.t. tx and k-th rx
                            Distance_rx_obj = self.distance_compute([tx_pos.x, tx_pos.y], [Rx[k].x, Rx[k].y],
                                                               [obj[j].x, obj[j].y])


                            # add the received signal              power,         carrier_freq,         n,              mu,              distance
                            sum[k] = sum[k] + self.received_signal(param.power,param.carrier_freq, param.sample_num, obj[j].absorption, Distance_rx_obj)
                #print(sum)

                for t in range(len(sum)):
                    Data[count,2 * t] = Decimal(abs(sum[t]))
                    # Data[count][2 * t] = Decimal(sum[t].real)
                    Data[count,2 * t + 1] = (Decimal(math.degrees(cmath.phase(sum[t])))+180)/360
                    # Data[count][2 * t + 1] = Decimal(sum[t].imag)

                # set the label
                labels[count,:] = [int(temp) for temp in val]
                # print('Iter: ' + str(count) + ' Labels: ' + str(labels[count][:]))
                count = count + 1

        print(a)
        # write data in text
        self.write_data(Data, labels, obj_coordinate, self.name)



    '''Generate the Codebook'''
    def gen_codebook(self, N_radi, N_theta):
        lst1 = list(itertools.product([0, 1], repeat=int(math.ceil(math.log(N_radi, 2)))))[
               1:]
        lst2 = list(itertools.product([0, 1], repeat=int(math.ceil(math.log(N_theta, 2)))))

        count = 0
        codebook = dict()
        for i in range(len(lst1)):
            a = list(map(str, lst1[i]))[0] + list(map(str, lst1[i]))[1]
            for j in range(len(lst2)):
                b = list(map(str, lst2[j]))[0] + list(map(str, lst2[j]))[1]
                codebook[count] = a + b
                count = count + 1

        return codebook

    ''' Generate Codebook for theta'''

    def gen_codebook_theta(self,N_theta):
        lst1 = list(itertools.product([0, 1], repeat=int(math.ceil(math.log(N_theta, 2)))))

        count = [[0, 90], [90, 180], [180, 270], [270, 355]]
        codebook_theta = dict()
        for i in range(len(lst1)):
            a = list(map(str, lst1[i]))[0] + list(map(str, lst1[i]))[1]
            codebook_theta[a] = count[i]

        return codebook_theta

    ''' Generate Codebook for radius'''
    def gen_codebook_radius(self,N_radi):
        lst1 = list(itertools.product([0, 1], repeat=int(math.ceil(math.log(N_radi, 2)))))[1:]
        count = [[0, 5], [5, 10], [10, 15]]
        codebook_radius = dict()
        for i in range(len(lst1)):
            a = list(map(str, lst1[i]))[0] + list(map(str, lst1[i]))[1]
            codebook_radius[a] = count[i]

        return codebook_radius

    '''Compute Distance between Object and Receiver'''
    def distance_compute(self,tx_pos, rx_pos, Obj_pos):
        return math.sqrt((Obj_pos[0] - tx_pos[0]) ** 2 + (Obj_pos[1] - tx_pos[1]) ** 2) + math.sqrt(
            (Obj_pos[0] - rx_pos[0]) ** 2 + (Obj_pos[1] - rx_pos[1]) ** 2)

    '''Compute Received Signal'''
    def received_signal(self, power,carrier_freq, n, mu, distance):
        c = 3e8
        signal = 0+ 1j*0
        amp = (power * mu / (distance ** 2))
        val =  2 * math.pi * carrier_freq * distance/c
        signal =   complex(amp*math.cos(val),amp*math.sin(val))
        return signal

    '''Write data into a .csv file'''
    def write_data(self, Data, labels, obj_coordinates, name):
        y=[]
        for i in range(int(Data.shape[1]/2)):
            y.append(Data[:,2*i])

        max_val = np.max(y)
        for i in range(int(Data.shape[1] / 2)):
            Data[:, 2 * i] = Data[:,2*i]/max_val

        np.savetxt("./Data500/" + name + "_labels.csv", labels, delimiter=',')
        np.savetxt("./Data500/" + name + "_data.csv", Data, delimiter=',')
        np.savetxt("./Data500/" + name + "_data_coord.csv", obj_coordinates, delimiter=',')


