import pyaudio
import numpy as np
import copy
import random
import math
import time
from scipy.signal import butter, lfilter


def NumToList(t):

        final_list = []
        for i in range(4):
            if(t%2 == 1):
                final_list.append('1')
            else:
                final_list.append('0')
            t = np.floor(t/2)
            
            
        print(final_list)
        
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
        #print("transmit",bits)

    def is_noise(self, signal):
        energy = np.sum(signal ** 2) / len(signal)  # Average energy

        #print("energy = ",energy)
        
        noise_threshold = 0.25  # Adjust this threshold based on testing
        return energy < noise_threshold
    def is_noise_frequency(self, signal):
        fft_values = np.fft.fft(signal)
        power = np.abs(fft_values) ** 2  # Power of each frequency bin
        average_power = np.mean(power)
        noise_power_threshold = 2000  # Adjust based on characteristics of your environment

        #print("averege power = ",average_power)

        return average_power < noise_power_threshold
    def is_noise_variability(self, signal):
        std_dev = np.std(signal)
        variability_threshold = 0.5  # Adjust based on testing
        #print("std_dev = ",std_dev)
        return std_dev < variability_threshold
    
    def is_noise_autocorrelation(self, signal):
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only one side
        significant_peak_threshold = 1000  # Adjust as necessary
        #print("corr = ",np.max(autocorr))
        return np.max(autocorr) < significant_peak_threshold

    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def is_noise_filtered(self, signal):
        filtered_signal = self.bandpass_filter(signal, self.f0 - 200, self.f1 + 200, self.sample_rate)
        #print("band_pass = ", np.mean(np.abs(filtered_signal)))
        return np.mean(np.abs(filtered_signal)) < 0.25 # Threshold for filtered signal
        


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


        check1 = self.is_noise(signal)
        check2 = self.is_noise_frequency(signal)
        check3 = self.is_noise_variability(signal)
        check4 = self.is_noise_autocorrelation(signal)
        check5 = self.is_noise_filtered(signal)


        if (check1 or check2 or check3 or check4 or check5 ):
            #print("Detected noise, no valid signal present.")
            bit = '2'  
        else:
            # Proceed with decoding the signal
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

        self.RTS_preamble = ['1','0','1','0']
        self.CTS_preamble = ['0','1','0','1']
        self.data_preamble = ['1','0','1','1']
        self.ACK_preamble = ['1','1','0','1']
        self.id = ['1','1']
        self.process_time = self.duration
        self.DIFS = 3
        self.SIFS= 2
        self.bit = '0'
        self.check_time = 0.1
        self.buffer = ['0','0','0','0']


    
    
    def send_RTS( self, reciver_id , data):
        
        time_req = np.ceil( (len(data) + 10)*self.duration + self.process_time )
        RTS = self.RTS_preamble +self.id + reciver_id + NumToList(time_req)

        #print("RTS sent = ",RTS)
        
        self.transmit(RTS)
            

    def recive_CTS(self,reciver_id):

        CTS_recived = 0
        
        buff = self.buffer

        for i in range(20):
            
            bit = self.read_signal()
            buff = buff[1 : ] + [bit]

            if ( buff == self.CTS_preamble):

                sender = [self.read_signal() , self.read_signal()]
                reciver = [self.read_signal() , self.read_signal()]
                wait_time = ListToNum([self.read_signal() ,self.read_signal(),self.read_signal(),self.read_signal()])

                if( sender == reciver_id and reciver == self.id ):
                    CTS_recived = 1
                    break
        
        return CTS_recived

    
    def CheckForAcg(self,reciver_id):

        ack_got = 0

        buff = self.buffer

        for i in range(20):
            
            bit = self.read_signal()
            buff = buff[1 : ] + [bit]

            if ( buff == self.ACK_preamble):
                sender = [self.read_signal() , self.read_signal()]
                reciver = [self.read_signal() , self.read_signal()]
                
                if( sender == reciver_id and reciver == self.id ):
                    ack_got = 1
                    break
        
        return ack_got
    
    def carrierSense(self):
        
        while(True):
                
            
            # buff =  self.buffer
            wait_time = 0

            x = 0
            
            for i in range(self.DIFS):
                bit = self.read_signal()
                # print(bit,i)
                if(bit=='1'):
                    break
                # buff = buff[1 : ] + [bit]

                    
                # if ( buff == self.RTS_preamble or buff == self.CTS_preamble):
                #     sender_id = [self.read_signal(),self.read_signal()]
                #     reciver_id = [self.read_signal(),self.read_signal()]
                    
                #     wait_time = ListToNum([self.read_signal() ,self.read_signal(),self.read_signal(),self.read_signal()])

                #     break
                # if (  buff == self.data_preamble or buff == self.ACK_preamble):
                #     sender_id = [self.read_signal(),self.read_signal()]
                #     reciver_id = [self.read_signal(),self.read_signal()]

                #     break
                # if( buff !=self.)
                x+=1 
                
            
            time.sleep(wait_time)

            if( x >= self.DIFS):
                break


    def send_data(self , data , reciver_id):
        """
        Sends the encoded data through the physical layer.

        Parameters:
            data (list of str): The data bits to be transmitted.
        """

        actual_data = self.data_preamble + [self.bit] +  NumToList(len(data))+ data
        t=1
        while(True):

            if(t!=1):
                time.sleep(random.randint(1,10)*self.DIFS*self.duration)
            
            print("hf")
            self.carrierSense()

            print("carrier sensed")

            self.send_RTS(reciver_id,data)
            print("RTS sent ")
            time.sleep(self.SIFS*self.duration)
            
            RTS_sent_succesful = self.recive_CTS(reciver_id)
            
            time.sleep(self.SIFS*self.duration)
            
            if(not RTS_sent_succesful):
                continue
            print("CTS recived")

            self.transmit(actual_data)

            print("data transmitted = ",actual_data)

            time.sleep(self.SIFS*self.duration)

            if(self.CheckForAcg(reciver_id) == 1):
                break
            t=t+1
            
        if ( self.bit == '0'):
            self.bit = '1'
        else:
            self.bit = '0'
            




dll_layer = DLL(sample_rate=44100,duration=0.25,f0=800,f1=1200,amplitute=1)

# message=list(input())
# start = time.time()
# while(time.time() - start < 5 ):
#     print(dll_layer.read_signal())
dll_layer.send_data(['1','0','1','0','1','0'],['0','1'])
