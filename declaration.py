import pyaudio
import numpy as np
import copy
import random
import math
import time
import threading

device = {
    '-1': ['1', '1', '1', '1'],
    '0': ['0', '0', '0', '1'],
    '1': ['0', '0', '1', '0'],
    '2': ['0', '1', '0', '0'],
    '3': ['1', '0', '0', '0']
}

def NumToList(t):

        final_list = []
        for i in range(4):
            if(t%2 == 1):
                final_list.append('1')
            else:
                final_list.append('0')
            t = np.floor(t/2)
        
        return final_list
    
def ListToNum(l):

    num = 0

    for i in range(4):
        if(l[i] == '1'):
            num += (1<<i)
    return num


class PhysicalLayer:

    # Constructor to intialize parameters for signal creation
    def __init__(self,sample_rate,duration,f0,f1,amplitute):
        
        """
        Initializes the PhysicalLayer class with the given parameters.
        
        Parameters:
            sample_rate (int): The sampling rate for the audio signals (samples per second).
            duration (float): The duration of each bit in seconds.
            f0 (float): The frequency representing bit '0' in Hz.
            f1 (float): The frequency representing bit '1' in Hz.
            amplitude (float): The amplitude of the audio signal.
        """

        self.sample_rate = sample_rate
        self.duration = duration
        self.f0 = f0
        self.f1 = f1
        self.amplitude = amplitute

        #intializing PyAudio interface
        self.port = pyaudio.PyAudio()

        #Openig an audio stream for both output(transmission) and input(reception)
        self.stream = self.port.open(format=pyaudio.paFloat32,
                                        channels=1,
                                        rate=self.sample_rate,
                                        output=True,
                                        input=True
                                    )
        
        self.transmit_lock = threading.lock()
        
    
    def __del__(self):

        """
        Cleans up the tesources by closing the audio stream and terminating the port      
        """

        self.stream.stop_stream()
        self.stream.close()
        self.port.terminate()

    
    def generate_signal(self,bit):
        """
        Generates an Audio signal corresponding to the given bit

        Parameters:
            bit (str): The bit('0' or '1')
        Returns:
            np.ndarray: The generated audio signal as Numpy array. 

        """

        # Select the frequency based on bit value
        frequency = self.f1 if (bit == '1') else self.f0

        # Create a time array based on the duration and sample rate
        time_arr = np.linspace(0, self.duration, int(self.duration*self.sample_rate),endpoint=False) 
        
        # Generate a sine wave signal at the chosen frequency
        signal = self.amplitude*np.sin( 2*np.pi*frequency*time_arr)
        return signal
    
    def transmit(self,bits):
        """
        Transmits a sequence of bits as audio signals.

        Parameters:
            bits (list of str): A list of bits ('0' or '1') to be transmitted.
        """
        
        with self.transmit_lock:

            # Iterate over each bit and generate its corresponding signal
            for bit in bits:
                signal = self.generate_signal(bit)
                # Write the generated signal to the audio output stream
                self.stream.write(signal.astype(np.float32).tobytes())
        
        
   
    def read_signal(self):
        """
        Reads an incoming audio signal and decodes it into a bit.

        Returns:
            str: The decoded bit ('0' or '1').
        """

        # Determine the number of samples to read based on the sample rate and duration
        numSamples = int(int(self.sample_rate*self.duration))
        
        # Read the raw audio data from the input stream
        rawData = self.stream.read(numSamples)

        # Convert the raw byte data to a NumPy array of floats
        signal = np.frombuffer(rawData,dtype = np.float32)

        bit=  self.decode_signal(signal)

        print(bit)

        return bit
    
    


    def decode_signal(self, signal):

            """
            Decodes a received audio signal into a corresponding bit.

            Parameters:
                signal (np.ndarray): The received audio signal as a NumPy array.

            Returns:
                str: The decoded bit ('0' or '1').
            """
            # Initialize an array to store the dominant bits determined from signal chunks
            bit_array = []
            num_chunks = 10  # Number of chunks to divide the signal into

            # Determine the size of each chunk
            chunk_size = int(len(signal) / num_chunks)


            # Iterate through the signal in chunks to analyze its frequency content
            for i in range(num_chunks):
                
                # Extract a chunk of the signal for analysis
                signal_chunk = signal[i*chunk_size: i*chunk_size + chunk_size]

                # Compute the Fast Fourier Transform (FFT) of the chunk to get frequency components
                fft_values = np.fft.fft(signal_chunk)

                # Get the corresponding frequency values for the FFT components
                frequencies = np.fft.fftfreq(len(signal_chunk), d=1/self.sample_rate)

                # Define the frequency ranges of interest around f0 and f1
                range_f0 = (self.f0 - 100, self.f0 + 100)
                range_f1 = (self.f1 - 100, self.f1 + 100)


                # Filter out the FFT values that fall within the range of f0

                indices_f0 = np.where((frequencies >= range_f0[0]) & (frequencies <= range_f0[1]))[0]
            
                # Filter out the FFT values that fall within the range of f1
                
                indices_f1 = np.where((frequencies >= range_f1[0]) & (frequencies <= range_f1[1]))[0]


                if np.max(np.abs(fft_values[indices_f0])) >= np.max(np.abs(fft_values[indices_f1])):
                    bit = 0  # Frequency nearer to f0 is dominant, representing bit '0'
                else:
                    bit = 1  # Frequency nearer to f1 is dominant, representing bit '1'

                
                # Append the determined bit to the bit array
                bit_array.append(bit)
            
            
            # Initialize sums for the first and second halves of the bit array
            firstHalfSum = 0
            secondHalfSum = 0

            # Initialize counters for the number of valid bits in each half
            len1 = 0
            len2 = 0

            # Decode the first half of the bit array (chunks 0 to 4)
            for bit in bit_array[0:5]:
                # if bit == -1:
                #     continue  # Skip invalid chunks (where no valid dominant frequency was found )
                
                len1 += 1
                firstHalfSum += bit

            # Decode the second half of the bit array (chunks 5 to 9)
            for bit in bit_array[5:10]:
                # if bit == -1:
                #     continue  # Skip invalid chunks (where no valid dominant frequency was found)
                
                len2 += 1
                secondHalfSum += bit

            # Calculate the total sum of bits and the total length of valid bits
            sum = firstHalfSum + secondHalfSum
            len_f = len1 + len2


            # Determine the final bit value based on the sum of decoded bits:
            # - If sum > len_f / 2: Most chunks have a dominant frequency nearer to f1, return '1'.
            # - If sum < len_f / 2: Most chunks have a dominant frequency nearer to f0, return '0'.
            if sum > len_f / 2:
                return '1'
            if sum < len_f / 2:
                return '0'

            # Special case: If sum equals len_f / 2, consider the second half as the tiebreaker.
            # - If the second half sum is less than half its length, return '0'.
            # - Otherwise, return '1'.
            if secondHalfSum < len2 / 2:
                return '0'
            else:
                return '1'


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
        self.bit = '0'
        self.b='2'
        self.time_len_bits = 4


        
        
        # Synchronization pattern to identify the start of a frame
        self.buffer = ['0','0','0','0','0','0']
        self.sync = ['0','0']
        
        self.len_RTS = len(self.sync) + len(self.RTS_preamble)+len(self.id)*2+ self.time_len_bits
        self.len_CTS = self.len_RTS
        self.len_ack = len(self.sync) + len(self.ACK_preamble)+len(self.id)*2
        self.wait_time = 0
        
        
    
    
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
        
        
        time.sleep(self.wait_time)
        self.wait_time = 0
        
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

        actual_data = self.sync + self.data_preamble + [self.bit] +  NumToList(len(data))+ data


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
            
        if ( self.bit == '0'):
            self.bit = '1'
        else:
            self.bit = '0'
            
    
    def send(self,messages):
        for message in messages:
            self.send_data(message[0],message[1])
            
            

    ## Reciver Part

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
                    self.wait_time = time1
                    time.sleep(time1)

    def read_data(self,sender):
        
        #print("READING DATA")
        last4bits=copy.deepcopy(self.buffer)
        final = []
        
        for _ in range(20):
            bit =   self.read_signal()
            last4bits = last4bits[1:]+[bit]

            if(last4bits == self.data_preamble):
                final=[self.read_signal()]
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
            if(data[0]!=self.b):
                print("data recived = " , data[1: ])
                self.b=data[0]
            time.sleep(self.SIFS*self.duration)
            self.send_ack(sender)

            print("ack sent")

            time.sleep(self.SIFS*self.duration)
    
    
        

  
dll_layer = DLL(sample_rate=44100,duration=0.25,f0=1000,f1=1200,amplitute=1)

data1 = [['1','0','1','0'],device['1']]

final_data = [data1]

t_send = threading.Thread(dll_layer.send,args=(data1,))
t_recieve = threading.Thread(dll_layer.recieve)


t_recieve.start()
t_send.start()

t_send.join()
t_recieve.join()


