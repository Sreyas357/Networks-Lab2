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
        self.broadcast = 0

    
    def send_ack(self,recvr_id):
        
        if self.broadcast:
            if self.id == device['2']:
                time.sleep((self.len_ack)*self.duration)
            elif self.id == device['3']:
                time.sleep((self.len_ack*2)*self.duration)
            elif self.id == device['1']:
                pass
        
        ack=[]
        ack =self.sync + self.ACK_preamble + self.id + recvr_id
        self.transmit(ack)
        
    
    def send_CTS( self, sender_id ,t):
        
        # RTS_sent = 0
        if self.broadcast:
            if self.id == device['2']:
                time.sleep((self.len_CTS)*self.duration)
            elif self.id == device['3']:
                time.sleep((self.len_CTS*2)*self.duration)
            elif self.id == device['1']:
                pass

        time.sleep(self.SIFS*self.duration)
        time_req = np.ceil( t  - self.len_RTS*self.duration )
        CTS = self.sync +self.CTS_preamble +self.id + sender_id + NumToList(time_req)
        print("CTS trans",CTS)
        
        self.transmit(CTS)

        if self.broadcast:
            if self.id == device['2']:
                time.sleep((self.len_CTS)*self.duration)
            elif self.id == device['3']:
                pass
            elif self.id == device['1']:
                time.sleep((self.len_CTS*2)*self.duration )
    
    def rec_rts(self):

        last4bits=copy.deepcopy(self.buffer)
        
        while(True):
            bit =   self.read_signal()
            last4bits = last4bits[1:]+[bit]

            if(last4bits == self.RTS_preamble): # RTS
                sender =[self.read_signal() for _ in range(len(self.id))]
                reciver =[self.read_signal() for _ in range(len(self.id))]
                time1=ListToNum([self.read_signal() for _ in range(4)])

                print("recived RTS from ", sender," to ",reciver)
                
                if(reciver == self.id or ( reciver == device['0'] and sender != self.id )):
                   
                    if (reciver == device['0'] ):
                       self.broadcast = 1

                    
                    print("Send CTS to " , sender)
                    self.send_CTS(sender,time1)
                    return sender
                else:
                    if (sender != self.id):
                        with open("a.txt", 'w') as file:
                            file.write(str(time1))
                    time.sleep(time1)

    def read_data(self,sender):
        
        #print("READING DATA")
        last4bits=copy.deepcopy(self.buffer)
        final = []
        
        for _ in range(20):
            bit =   self.read_signal()
            last4bits = last4bits[1:]+[bit]

            if(last4bits == self.data_preamble):
                final=[]
                length=ListToNum([self.read_signal() ,self.read_signal() ,self.read_signal() ,self.read_signal() ])
                
                print("lenght = ",length)

                for _ in range(length):
                    final += [self.read_signal()]
                return final
            
        
        return final
            
                
                    
    def recieve(self):
        """
        Receives data from the physical layer by detecting the synchronization pattern
        and then reading and decoding the data bits.

        Returns:
            list of str: The received and corrected data bits.
        """
        while(True):
            
            time.sleep(self.SIFS*self.duration)
            
            self.broadcast = 0
            
            sender=self.rec_rts()
            
            time.sleep(self.SIFS*self.duration)

            data=self.read_data(sender)
            
            if(data==[]):
                continue
            
            print("data recived = " , data[1: ])
            
            time.sleep(self.SIFS*self.duration)
            
            self.send_ack(sender)
            print("ack sent")

            time.sleep(self.SIFS*self.duration)
        

    






    

dll_layer = DLL(sample_rate=44100,duration=0.25,f0=1000,f1=1200,amplitute=1)


# for _ in range(30):
#     (dll_layer.read_signal())

dll_layer.recieve()
