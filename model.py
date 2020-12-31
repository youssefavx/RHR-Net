#Rough sketch - work in progress

from tensorflow import keras 
import keras 
model = Sequential() #Is "Sequential" even right? Do I have to specify it's some kind of bi-directional RNN?

#First 6 GRU Layers are currently NOT bidirectional which they have in their paper
gru_layer_1 = keras.layers.GRU(2) #Is there some parameter to specify timesteps...whatever those are?
gru_layer_2 = keras.layers.GRU(128) 
gru_layer_3 = keras.layers.GRU(256) 
gru_layer_4 = keras.layers.GRU(512) 
gru_layer_5 = keras.layers.GRU(256) 
gru_layer_6 = keras.layers.GRU(128) 
gru_layer_7 = keras.layers.GRU(1) 

model.add(gru_layer_1) 
model.add(gru_layer_2) 
model.add(gru_layer_3) 
model.add(gru_layer_4) 
model.add(gru_layer_5) 
model.add(gru_layer_6) 
model.add(gru_layer_7)

#model.compile(loss='logcosh', optimizer='RMSprop', metrics=['accuracy']) #Something like this? Not sure if this is how they set it up and if metrics was set to 'accuracy'

#Wav file:
#How does the model take in data? Does a WAV file have to be reformatted to something?
#I know it's possible read wave files using scipy like so, but would the samples just be fed into the first layer?
from scipy.io import wavfile
import scipy.io

audio = wavfile.read(filename)
samplerate = audio[0]
data = audio[1]
#Get the first 100 samples:
data[:100]


def length(audio, number_of_samples=1412):
  samplerate, data = audio
    if number_of_samples is not None:
      return data[:number_of_samples].shape[0] / samplerate
    else:
      return data.shape[0] / samplerate
      
#If your wave file is 22kHz, 1412 will give you an approximation of 64 ms (the chunk length they use in their paper):
#length(audio, number_of_samples=1412)
#Will output: 0.06403628117913832

#For 16kHz (the rate they use in their paper):
#length(audio, number_of_samples=1024)
#Will output: 0.064

#To calculate the number of samples you'd need for a particular sample rate:
#milliseconds * sample rate = number of samples
#64 * 44.1 = 2822.4
#So you might round up or down.

#Now the question is, how do you feed in the data, do you just feed in the first 1024 samples, then the next to the first Keras layer? 
#Are you just giving it noisy signals or noisy + clean?
#Is that what the 2 units of the first layer are for, or is that the function of the bidirectional RNN?

