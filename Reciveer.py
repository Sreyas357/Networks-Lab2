import pyaudio
import numpy as np
import copy
import random
import math
import time

# Base class representing the Physical Layer in a communication system
def NumToList(t):

        final_list = []
        for i in range(4):
            if(t%2 == 1):
                final_list.append('1')
            else:
                final_list.append('0')
            t /= 2
        
        return final_list
    
def listtoNum(l):

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




def xor(a,b):
    """
    Performs a bitwise XOR operation between two binary strings.

    Parameters:
        a (str): The first binary string.
        b (str): The second binary string.

    Returns:
        str: The result of the XOR operation as a binary string.
    """

    ans = ""
    for i in range(len(a)):
        if( a[i]  != b[i]):
            ans += '1'
        else:
            ans += '0'
    return ans

def remainder(a, b):
    """
    Calculates the remainder of the binary division of 'a' by 'b', used in CRC.

    Parameters:
        a(str) : The binary dividend.
        b(str) : The binary divisor (CRC polynomial).

    Returns:
        str: The remainder after the division.
    """

    w =  a[:len(b) - 1]
    for i in range(len(a) - len(b)+1):
        w += a[i + len(b) - 1]
        
        # Determine whether to XOR with '0' or 'b' based on the leading bit
        if(w[0] == '0'):
            y = ['0'] * len(b)
            x = ''.join(y)
        else:
            x = b
        
        # Update the working string with the XOR result, removing the leading bit
        w = xor(w,x)[1:]
    
    return w


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
        self.id=['0','1']
        self.b='2'
        self.check_time=0.25
        # Synchronization pattern to identify the start of a frame
    def send_ack(self,reci_id):
        ack=[]
        ack+=['1','1']
        ack+=self.id
        ack+=reci_id
        return ack
    def send_CTS( self, reciver_id ,t):
        time_req = t + self.process_time  - 10*self.duration
        RTS = self.RTS_preamble +self.id + reciver_id + NumToList(time_req)
    def rec_rts(self):
        last4bits=['2','2']
        while(True):
            bit =   self.read_signal()
            last4bits = last4bits[1:]+[bit]
            if(last4bits == ['0','0']): # RTS
                sender =[self.read_signal(),self.read_signal()]
                reciver =[self.read_signal(),self.read_signal()]
                time=listtoNum([self.read_signal() for _ in range(4)])
                if(reciver == self.id):
                    self.send_CTS(self,sender,time)
                    return sender
                else:
                    time.sleep(time)
    def read_data(self,sender):
        last4bits=['2','2']
        final = []
        start_time=time.time
        while(time.time-start_time<self.check_time):
            bit =   self.read_signal()
            last4bits = last4bits[1:]+[bit]
            if(last4bits == ['1','0']): # RTS
                final=[self.read_signal()]
                length=listtoNum([self.read_signal() for _ in range(4)])
                for i in range(length):
                    final.append(self.read_signal())
                    return final
            elif(last4bits[0]!='2' and last4bits[1]!='2'):
                break
        return final
            
                
                    
    def recieve(self):
        """
        Receives data from the physical layer by detecting the synchronization pattern
        and then reading and decoding the data bits.

        Returns:
            list of str: The received and corrected data bits.
        """
        while(True):
            sender=self.rec_rts()
            data=self.read_data(sender)
            if(data==[]):
                continue
            if(data[0]!=self.b):
                print(data[1: ])
                self.b=data[0]
        

    def finalData(self,data):
        """
        Prepares the final data to be transmitted by adding synchronization bits
        and the length of the data.

        Parameters:
            data (list of str): The data bits to be transmitted.

        Returns:
            list of str: The final data bits ready for transmission.
        """

        # Encode the data with CRC
        finalCodeWord = self.encode(data)

        # Calculate the length of the encoded data
        l = len(finalCodeWord)

        # Start with the synchronization pattern
        final_data = copy.deepcopy(self.syncBits)

        # Append the length information (5 bits)
        for i in range(5):
            final_data.append(f'{l%2}')
            l = int(l/2)
        
        # Append the Final CodeWord
        final_data += finalCodeWord
        return final_data


    def send_data(self , data):
        """
        Sends the encoded data through the physical layer.

        Parameters:
            data (list of str): The data bits to be transmitted.
        """
        self.transmit(data)
    

dll_layer = DLL(sample_rate=44100,duration=0.25,f0=800,f1=1200,amplitute=1)
# sender part
def flip(data ,index):
    if(data[index] == '0'):
        data[index] = '1'
    else:
        data[index] = '0'


message = list(input())
finalData = dll_layer.finalData(message)

a_and_b = input().split(' ')
a = float(a_and_b[0])
b = float(a_and_b[1])

pre_len = len(dll_layer.syncBits)+5

error_index1 = pre_len+math.ceil((len(message)+dll_layer.crc_degree)*a)-1
error_index2 = pre_len+math.ceil((len(message)+dll_layer.crc_degree)*b)-1

flip(finalData,error_index1)
if b != 0:
    flip(finalData,error_index2)

print(finalData)
dll_layer.send_data(finalData)
