import pyaudio
import numpy as np
import copy
import random
import math
import time


def NumToList(t):

        final_list = []
        for i in range(4):
            if(t%2 == 1):
                final_list.append('1')
            else:
                final_list.append('0')
            t /= 2
        
        return final_list
    
def ListToNum(l):

    num = 0

    for i in range(4):
        if(l[i] == '1'):
            num += (1<<i)
    return num


# Base class representing the Physical Layer in a communication system
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

        # Decode the signal to determine whether it represents a '0' or '1'
        bit = self.decode_signal(signal)

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
        for i in range(0, int(self.duration * self.sample_rate), int(chunk_size)):
            
            # Extract a chunk of the signal for analysis
            signal_chunk = signal[i: i + chunk_size]

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

            # If no relevant frequencies are found, append -1 to indicate no valid detection
            if len(indices_f0) + len(indices_f1) == 0:
                bit_array.append(-1)
                continue

            # Determine the final bit by comparing which range has a stronger signal
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
            if bit == -1:
                continue  # Skip invalid chunks (where no valid dominant frequency was found )
            
            len1 += 1
            firstHalfSum += bit

        # Decode the second half of the bit array (chunks 5 to 9)
        for bit in bit_array[5:10]:
            if bit == -1:
                continue  # Skip invalid chunks (where no valid dominant frequency was found)
            
            len2 += 1
            secondHalfSum += bit

        # Calculate the total sum of bits and the total length of valid bits
        sum = firstHalfSum + secondHalfSum
        len_f = len1 + len2


        if len_f == 0:
            return '2'

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

        self.RTS_preamble = ['0','0']
        self.CTS_preamble = ['0','1']
        self.data_preamble = ['1','0']
        self.ACK_preamble = ['1','1']
        self.id = ['1','0']
        self.process_time = 0.5
        self.DIFS = 0.25
        self.SIFS= 0.125
        self.bit = '0'
        self.check_time = 0.1


    
    
    def send_RTS( self, reciver_id , data):
        
        RTS_sent = 0
        time_req = np.ceil( (len(data) + 10)*self.duration + self.process_time )
        RTS = self.RTS_preamble +self.id + reciver_id + NumToList(time_req)
        
        self.transmit(RTS)
    
    def send_CTS( self, reciver_id ,t):
        
        RTS_sent = 0
        time_req = t + self.process_time  - 10*self.duration
        RTS = self.RTS_preamble +self.id + reciver_id + NumToList(time_req)
        
        self.transmit(RTS)
            

    def recive_CTS(self,reciver_id):

        CTS_recived = 0
        start_time = time.time()
        buff = ['2','2']

        while time.time()-start_time < self.check_time:
            
 
            bit = self.read_signal()
            buff = buff[1 : ] + [bit]

            if ( buff == self.CTS_preamble):
                sender = [self.read_signal() , self.read_signal()]
                reciver = [self.read_signal() , self.read_signal()]
                if( sender == reciver_id and reciver == self.id ):
                    CTS_recived = 1
                    break
        return CTS_recived

    
    def CheckForAcg(self,reciver_id):

        ack_got = 0

        start_time = time.time()
        buff = ['2','2']

        while time.time()-start_time < self.check_time:
            
            
            bit = self.read_signal()
            buff = buff[1 : ] + [bit]

            if ( buff == self.ACK_preamble):
                sender = [self.read_signal() , self.read_signal()]
                reciver = [self.read_signal() , self.read_signal()]
                if( sender == reciver_id and reciver == self.id ):
                    ack_got = 1
        
        return ack_got
    
    def carrierSense(self):
        
        while(True):
                
            
            buff =  ['2','2']
            wait_time = 0

            start_time = time.time()
            present_time = start_time

            while( present_time-start_time < self.DIFS ):

                bit = self.read_signal()
                buff = buff[1 : ] + [bit]

                if ( buff[0] != '2' and buff[1] != '2' ):
                    
                    if ( buff == self.RTS_preamble or buff == self.CTS_preamble):
                        sender_id = [self.read_signal(),self.read_signal()]
                        reciver_id = [self.read_signal(),self.read_signal()]
                        wait_time = ListToNum([self.read_signal() ,self.read_signal(),self.read_signal(),self.read_signal()])
                    break

                present_time = time.time()
            
            
            time.sleep(wait_time)

            if( present_time-start_time >= self.DIFS):
                break



    def send_data(self , data , reciver_id):
        """
        Sends the encoded data through the physical layer.

        Parameters:
            data (list of str): The data bits to be transmitted.
        """

        actual_data = self.data_preamble + self.id + reciver_id + NumToList(len(data)) + [self.bit] + data



        while(True):
            
            time.sleep(random.randint(1,5)*self.DIFS)
            
            self.carrierSense()
            
            RTS_sent_succesful = 0

            while(not RTS_sent_succesful):
                self.send_RTS(reciver_id,data)
                time.sleep(self.SIFS)
                RTS_sent_succesful = self.recive_CTS(reciver_id)
            
            self.transmit(actual_data)

            time.sleep(self.SIFS)

            if(self.CheckForAcg(reciver_id) == 1):
                break
        

        if ( self.bit == '0'):
            self.bit = '1'
        else:
            self.bit = '0'
            




dll_layer = DLL(sample_rate=44100,duration=0.25,f0=800,f1=1200,amplitute=1)

# message=list(input())
dll_layer.send_data(['1','0','1','0'],['0','1'])
