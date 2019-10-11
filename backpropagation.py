import math
import random

#calculate random number between [a,b)
def randomNumber(a, b):
    return (b-a)*random.random() + a

#constract the matrix
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

#sigmoid function
def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

#derivative of sigmoid function
def dsigmoid(y):
    return (y)*(1.0-y)

class ANN:
    def __init__(self, ni, nh, no):
        #number of input, hidden, and output
        self.ni = ni + 1
        self.nh = nh + 1
        self.no = no

        #activations for input, hidden, and output
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        #create weights with initial value = 0.0
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        #set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = randomNumber(-0.05, 0.05)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = randomNumber(-0.05, 0.05)
		#last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def feedForward(self, inputs):
        #input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        #hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        #output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backpropagation(self, targets, learningRate, momentum):
        #calculate error for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        #calculate error for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        #update output weights
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = self.wo[j][k] + learningRate*output_deltas[k]*self.ah[j] + momentum*self.co[j][k]
                self.co[j][k] = output_deltas[k]*self.ah[j]

        #update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = self.wi[i][j] + learningRate*hidden_deltas[j]*self.ai[i] + momentum*self.ci[i][j]
                self.ci[i][j] = hidden_deltas[j]*self.ai[i]

        #calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, example, iterations=100000, learningRate=0.1, momentum=0.1):
        for i in range(iterations):
            error = 0.0
            for j in example:
                inputs = j[0]
                targets = j[1]
                self.feedForward(inputs)
                error = error + self.backpropagation(targets, learningRate, momentum)
            if i % 100 == 0:
                print('E= %-.4f' % error)
				
    def test(self, example):
        for i in example:
            print(i[0], ': outputs -->', self.feedForward(i[0]))

def myMain():
	trainExamples = [
        [[0,0], [0]],
        [[0,1], [0]],
        [[1,0], [0]],
        [[1,1], [1]]
	]
	
	x = ANN(2, 1, 1)
	x.train(trainExamples)
	print()
	x.test(trainExamples)
	print()
	x.weights()

myMain()