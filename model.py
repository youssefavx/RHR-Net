#Rough sketch - work in progress

#What is left to implement?
#1. Learning rate: starting at 10^-4 and gradually decreasing to 10^-8 during training. Is that done through a kind of 'learning rate schduling' on Keras?
#2. Activation function: PReLU (but it isn't clear if this is applied for only certain layers with residual connections, or for all layers)
#3. Residual connections between certain layers
#4. Initializer: Xavier normal initializer ( "We use the Xavier normal initializer [34] for the kernel weights, with zero-initialized biases." Is this it: https://keras.io/api/layers/initializers/#glorotnormal-class ?) - Zero-initalizaed biases? Kernel weights? Is that different to weights?)
#   Maybe you can do this with something like: keras.layers.GRU(2, kernel_initializer='glorot_normal', bias_initializer='zeros') #BUT... is it applied to all layers? (I'm assuming yes)
#5. Initializer of recurrent states: "The initializer for the recurrent states is a random orthogonal matrix [35], which helps the RNN stabilize by avoiding vanishing or exploding gradients. The stability occurs because the orthogonal matrix has an absolute eigenvalue of one, which avoids the gradients from exploding or vanishing due to repeated matrix multiplication." ... huh? Is this something I have to set on Keras? (random orthogonal matrix?)
#   Key terms:
#   'Recurrent state' what does that refer to, the GRU units?
#   'Random orthogonal matrix' ... is there some parameter I need to set with this, how is this implemented?
#6. Making GRU Layers Bi-directional 

#All questions so far:

#GRU
#1. How do you make Bi-directional GRU Layers?
#2. Do I have my GRU layer parameters set up correctly, is this right for ALL layers: keras.layers.GRU(2, kernel_initializer='glorot_normal', bias_initializer='zeros') 

#Data
#1. How do you input data, raw samples? If so, how? Meaning in the first layer, or during training
#2. Is the data coming in only going to be noisy audio through the first layer? How do you input 2 streams, noisy and clean?
#3. If I want to switch to 22kHz, or 44.1kHz, do the GRU layer units need to change in size relative to that?
#

from tensorflow import keras 
import keras 
model = Sequential() #Is "Sequential" even right? Do I have to specify it's some kind of bi-directional RNN?

#First 6 GRU Layers are currently NOT bidirectional which they have in their paper
gru_layer_1 = keras.layers.GRU(2) #I assume timesteps == samples in this case? 
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

