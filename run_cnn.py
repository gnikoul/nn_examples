from layers import CNN, AVRPOLL, Sigmoid, shape, FC, flatten, Tanh, softmax
from numpy import random


n_feature_maps = 6
receptive_field_size = [5,5]
step = 1

class lenet():
    def __init__(self):
        # Feature extraction
        self.cnn1 = CNN(n_input_ch=1, n_feature_maps=6, receptive_field_size=[5,5], step=1)
        self.avrpool2 = AVRPOLL(n_feature_maps=6, receptive_field_size=[2,2], step=2)
        self.act3 = Sigmoid()

        self.cnn4 = CNN(n_input_ch=6, n_feature_maps=16, receptive_field_size=[5,5], step=1)
        self.avrpool5 = AVRPOLL(n_feature_maps=16, receptive_field_size=[2,2], step=2)
        self.act6 = Sigmoid()

        self.cnn7 = CNN(n_input_ch=16, n_feature_maps=120, receptive_field_size=[5,5], step=1)

        # classification
        self.fc8 = FC(48000, 84)
        self.act9 = Tanh()
        self.fc10 = FC(84, 10)


    def feature_extraction(self, input, prints = False):
        if prints: print(shape(input))
        out = self.cnn1.forward(input)
        if prints: print(shape(out))
        out = self.avrpool2.forward(out)
        out = self.act3.forward(out)
        if prints: print(shape(out))

        out = self.cnn4.forward(out)
        if prints: print(shape(out))
        out = self.avrpool5.forward(out)
        out = self.act6.forward(out)
        if prints: print(shape(out))

        out = self.cnn7.forward(out)
        if prints: print(shape(out))

        return out
    
    def classification(self, input):
        out = self.fc8.forward(input)
        out = self.act9.forward(out)
        out = self.fc10.forward(out)

        return out
    
    def forward(self, input):
        out = self.feature_extraction(input, prints=False)
        out = flatten(out)
        out = self.classification(out)

        return out

ln = lenet()
input = random.uniform(low=-10,high=10,size=(32,32,1)).tolist()
out = ln.forward(input)
print(softmax(out))