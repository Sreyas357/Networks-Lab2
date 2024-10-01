import pyaudio
import numpy as np
import copy
import random
import math
import time


from declaration import PhysicalLayer,device,NumToList,ListToNum


# Data Link Layer class extending the Physical Layer
class DLL(PhysicalLayer):

    def __init__(self,sample_rate,duration,f0,f1,amplitute):
        """
        Initializes the DLL class, which extends the PhysicalLayer with additional
        data link layer functionalities such as CRC error detection and synchronization.

        Parameters:
            sample_rate (int): The sampling rate for audio signals (samples per second).
            duration (float): The duration of each bit in seconds.
            f0 (float): The frequency representing bit '0' in Hz.
            f1 (float): The frequency representing bit '1' in Hz.
            amplitude (float): The amplitude of the audio signal.
        """
        # Call the base class (PhysicalLayer) constructor
        PhysicalLayer.__init__(self,sample_rate,duration,f0,f1,amplitute)

        PhysicalLayer.__init__(self,sample_rate,duration,f0,f1,amplitute)
        self.RTS_preamble = ['1','1','0','0','1','1']
        self.CTS_preamble = ['0','1','1','1','0','1']
        self.data_preamble = ['1','0','1','1','0','0']
        self.ACK_preamble = ['1','0','0','1','1','0']
        self.id = device['3']
        self.process_time = self.duration
        self.DIFS = 3
        self.SIFS= 1
        
        self.time_len_bits = 4
        
        # Synchronization pattern to identify the start of a frame
        self.buffer = ['0','0','0','0','0','0']
        self.sync = ['0','0']
        
        self.len_RTS = len(self.sync) + len(self.RTS_preamble)+len(self.id)*2+ self.time_len_bits
        self.len_CTS = self.len_RTS
        self.len_ack = len(self.sync) + len(self.ACK_preamble)+len(self.id)*2
        
    
    
    def send_RTS( self, reciver_id , data):
        
        if reciver_id != device['0']:
            time_req = np.ceil( (len(data) + self.len_RTS + self.len_CTS + self.len_ack )*self.duration  + (self.SIFS*4)*self.duration + 1 )
        else:
            time_req = np.ceil( (len(data) + self.len_RTS + self.len_CTS * 2 + self.len_ack * 2 )*self.duration  + (self.SIFS*8)*self.duration + 1 )

        RTS = self.sync +self.RTS_preamble +self.id + reciver_id + NumToList(time_req)
        
        self.transmit(RTS)
            

    def recive_CTS(self,reciver_id):

        CTS_recived = 0
        
        buff = copy.deepcopy(self.buffer)

        for i in range(20):
            
            bit = self.read_signal()
            buff = buff[1 : ] + [bit]

            if ( buff == self.CTS_preamble):

                sender = [self.read_signal() for _ in range(len(self.id))]
                reciver = [self.read_signal() for _ in range(len(self.id))]
                wait_time = ListToNum([self.read_signal() ,self.read_signal(),self.read_signal(),self.read_signal()])

                if(  sender == reciver_id and reciver == self.id ):
                    CTS_recived = 2
                    break
                else:
                    time.sleep(wait_time)
        
        return CTS_recived

    
    def CheckForAcg(self,reciver_id):

        ack_got = 0

        buff = copy.deepcopy(self.buffer)

        for i in range(20):
            
            bit = self.read_signal()
            buff = buff[1 : ] + [bit]

            if ( buff == self.ACK_preamble):
                sender = [self.read_signal() for _ in range(len(self.id))]
                reciver = [self.read_signal() for _ in range(len(self.id))]
                
                if( sender == reciver_id and reciver == self.id ):
                    ack_got = 2
                    break
        
        return ack_got
    
    def carrierSense(self):
        
        wait_time = self.read_wait_time() 
        time.sleep(wait_time)
        
        for _ in range(self.DIFS*3):
            bit = self.read_signal()
            if(bit=='1'):
                time.sleep(1)
                return 0
            
        return 1
                


    def send_data(self , data , reciver_id):
        """
        Sends the encoded data through the physical layer.

        Parameters:
            data (list of str): The data bits to be transmitted.
        """

        actual_data = self.sync + self.data_preamble  +  NumToList(len(data))+ data


        while(True):

            # time.sleep(random.randint(1,10)*self.DIFS*self.duration)
            if self.id == device['2']:
                time.sleep(5*self.DIFS*self.duration)
            elif self.id == device['3']:
                time.sleep(10*self.DIFS*self.duration)
            elif self.id == device['1']:
                pass
            
            while ( not self.carrierSense() ):
                pass

            print("carrier sensed")

            self.send_RTS(reciver_id,actual_data)
            print("RTS sent ")
            time.sleep(self.SIFS*self.duration)
            
            RTS_sent_succesful = 0
            if reciver_id != device['0']:
                RTS_sent_succesful = self.recive_CTS(reciver_id)
                time.sleep(self.SIFS*self.duration)
            else:
                if self.id == device['1']:
                    
                    time.sleep((self.len_CTS)*self.duration)
                    RTS_sent_succesful += self.recive_CTS(device['2'])/2
                    time.sleep(self.SIFS*self.duration)
                    RTS_sent_succesful += self.recive_CTS(device['3'])/2
                    time.sleep(self.SIFS*self.duration)
                
                if self.id == device['2']:
                    
                    RTS_sent_succesful += self.recive_CTS(device['1'])/2
                    time.sleep((self.len_CTS)*self.duration)
                    RTS_sent_succesful += self.recive_CTS(device['3'])/2
                    time.sleep(self.SIFS*self.duration)
                    
                if self.id == device['3']:
                    
                    RTS_sent_succesful += self.recive_CTS(device['1'])/2
                    time.sleep(self.SIFS*self.duration)
                    RTS_sent_succesful += self.recive_CTS(device['2'])/2
                    time.sleep(self.SIFS*self.duration)
                    time.sleep((self.len_CTS)*self.duration)
            
            
            
            if( RTS_sent_succesful != 2):
                continue

            print("CTS recived")

            self.transmit(actual_data)

            print("data transmitted = ",actual_data)

            time.sleep(self.SIFS*self.duration)
            
            ACK_recived = 0
            if reciver_id != device['0']:
                ACK_recived = self.CheckForAcg(reciver_id)
                time.sleep(self.SIFS*self.duration)
            else:
                if self.id == device['1']:
                    
                    time.sleep((self.len_ack)*self.duration)
                    ACK_recived += self.CheckForAcg(device['2'])/2
                    time.sleep(self.SIFS*self.duration)
                    ACK_recived += self.CheckForAcg(device['3'])/2
                    time.sleep(self.SIFS*self.duration)
                
                if self.id == device['2']:
                    
                    ACK_recived += self.CheckForAcg(device['1'])/2
                    time.sleep((self.len_ack)*self.duration)
                    ACK_recived += self.CheckForAcg(device['3'])/2
                    time.sleep(self.SIFS*self.duration)
                    
                if self.id == device['3']:
                    
                    ACK_recived += self.CheckForAcg(device['1'])/2
                    time.sleep(self.SIFS*self.duration)
                    ACK_recived += self.CheckForAcg(device['2'])/2
                    time.sleep(self.SIFS*self.duration)
                    time.sleep((self.len_ack)*self.duration)
            
            
            if(ACK_recived == 2):
                break
            
            




dll_layer = DLL(sample_rate=44100,duration=0.25,f0=1000,f1=1200,amplitute=1)

# msg_and_dest_1 = input("Enter Message and Destination 1: ")
# msg1, dest1 = msg_and_dest_1.split(' ')
# msg1 = list(msg1)
# dest1 = device[dest1]
# msg_and_dest_2 = input("Enter Message and Destination 2: ")
# msg2, dest2 = msg_and_dest_2.split(' ')
# msg2 = list(msg2)
# dest2 = device[dest2]

# inp = input("Enter something to send the first message: ")
# if inp:
#     if dest1 == ['1', '1', '1', '1']:
#         pass
#     else:
#         dll_layer.send_data(msg1, dest1)

# inp = input("Enter something to send the second message: ")
# if inp:
#     if dest2 == ['1', '1', '1', '1']:
#         pass
#     else:
#         dll_layer.send_data(msg2, dest2)

# dll_layer.send_data(['1','0','1','0','1','0','0','1','1','1'],device['1'])
dll_layer.send_data(['1','1','1','0','0','1','0','1','0'],device['0'])