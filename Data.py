''' Class Project Machine Learning: 5523
Authors: Rupam Kundu and Tanmoy Das
Project: Object detection using WiFi
This code: Generating the Data'''

from Data_generate import Data_generate
import numpy as np
import sys

class param():
    def __init__(self):
        self.N_obj=0
        self.N_repeat = 0
        self.N_sector = 0
        self.Absorption = []
        self.N_theta =0
        self.Tx_Coord = 0
        self.N_radi = 0
        self.Rx_Coord =[]
        self.N_rx =0
        self.power = 0
        self.carrier_freq=0
        self.sample_num = 0

def run():
    # Define the set of params: Training Data
    parameter = param()
    parameter.N_obj = 10 # Set Total Number of Objects
    parameter.N_sector = 12 # set the number of sectors
    parameter.N_repeat = int(sys.argv[2]) #No of data points will be N_repeat* 2**N_sector
    parameter.Absorption = [1, 1, 1, 1, 1, 1] # absorption parameter
    parameter.N_theta = 4 # Number of theta partition
    parameter.N_radi=3 #no of concentric circles
    # Set Receiver's Position
    lst=[]
    for i in range(50):
        lst.append([-15*np.random.rand(),15*np.random.rand()])
    
    #Setting1: np.array([[-10, 6], [2, 10], [-5, -5]])
    #setting2: np.array([[-15, 0], [5, 10], [-5, 11], [10,-12], [-5,-10], [8,9]])  # in meters
    #setting3:lst

    #parameter.Rx_Coord = np.array([[-10, 6], [2, 10], [-5, -5]])
    #parameter.Rx_Coord = np.array([[-10, 6], [2, 10], [-5, -5], [2, -5], [1, 1], [0, 15], [0, -15]])

    #parameter.Rx_Coord = np.array([[-10, 6], [2, 10], [-5, -5]])
    parameter.Rx_Coord = np.array([[-10, 6], [2, 10], [-5, -5], [2, -5], [1, 1], [0, 15], [0, -15]])
    #parameter.Rx_Coord = np.array([[-10, 6], [2, 10], [-5, -5], [2, -5], [1, 1], [0, 15], [0, -15], [10, 0], [-10, 0], [-1, -1], [0, 0]])
    parameter.Tx_Coord = [0, 0] #tx coordinates
    parameter.N_rx = len(parameter.Rx_Coord)
    # Set signal parameters
    parameter.power =1
    parameter.carrier_freq = 5.9e9
    parameter.sample_num=1

    print('-----------------------------------------------------------------------------')
    print("Generating " + sys.argv[1] + "_data  and " + sys.argv[1] + "_labels")
    print('-----------------------------------------------------------------------------')
    #Run the data generate
    Data_val = Data_generate(sys.argv[1])
    Data_val.generate_data(parameter)

    print('\nDone')









if __name__ == "__main__":
    run()
