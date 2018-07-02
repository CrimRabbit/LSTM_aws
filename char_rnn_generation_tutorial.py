from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import unicodedata
import string
import torch.optim as optim

all_letters = string.ascii_letters + string.punctuation + string.digits + " "
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    category_lines = []
    with open(filename,'r') as f:
        for line in f:
            #default code splits commas, but commas exist in text too
            #added a replace <~> before split, then replace back after
            v=line.strip().replace('=','').replace('/',' ').replace('+',' ').replace('(',' ').replace('[',' ').replace(')',' ').replace(']',' ').replace(', ','<~>').split(',')
            for w in v:
                w=w.replace('<~>',',')
                if (len(w)>1): #(w not in filterwords) and 
                    category_lines.append(w)
    return category_lines

# Build the category_lines dictionary, a list of lines per category
all_categories = []

all_categories = readLines('star_trek_transcripts_all_episodes_f.csv')
all_categories = all_categories[0:500] #reducing size to test
np.random.shuffle(all_categories)
n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found.')

import torch
import torch.nn as nn

#hidden is memory layer
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_letters = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=0.1)
        self.fc = nn.Linear(hidden_size,self.num_letters)

        #self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        #self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        #self.o2o = nn.Linear(hidden_size + output_size, output_size)
        #self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.temperature = 1

    def changeTemperature(self, temp):
        self.temperature = temp

    def forward(self, x, hidden):
        output,(h_n,c_n) = self.lstm(x,hidden)
        output = self.fc(output.squeeze(1))
        output = self.softmax(output/self.temperature)
        return output, (h_n,c_n)

    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),torch.zeros(self.num_layers, 1, self.hidden_size))

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    idx = all_letters.find(letter)
    if (idx==-1):
        print('cannot find letter')
        print(letter)
    return idx

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(model, data, criterion, optimizer):
    #data = data[0:100]
    running_loss = 0
    running_correct = 0
    running_count = 0
    #target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()
    model.train()

    for idx, line in enumerate(data, start=1):
        model.zero_grad()
        newLine = inputTensor(line)
        #print(line)
        target = targetTensor(line)
        #print(target)
        
        output,_ = model(newLine,hidden)
        loss = criterion(output,target)
        
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss
        _,pred = output.max(1)
        running_correct += torch.eq(pred,target).sum().item()
        running_count += len(line)

    epoch_loss = running_loss/len(data)
    epoch_acc = running_correct/running_count
    
    print('Train: {}/{} (acc: {:.4f}), loss: {:.4f}'.format(running_correct, running_count, epoch_acc, epoch_loss))
    
    return epoch_loss, epoch_acc

def test(model, data, criterion):
    running_loss = 0
    running_correct = 0
    running_count = 0
    
    model.eval()
    hidden = model.initHidden()
    
    for idx, line in enumerate(data, start=1):
        #model.zero_grad()
        newLine = inputTensor(line)
        #print(line)
        target = targetTensor(line)
        #print(target)
        
        output,_ = model(newLine,hidden)
        loss = criterion(output,target)
        
        #loss.backward()
        #optimizer.step()
        
        
        running_loss += loss
        _,pred = output.max(1)
        running_correct += torch.eq(pred,target).sum().item()
        running_count += len(line)

    epoch_loss = running_loss/len(data)
    epoch_acc = running_correct/running_count
    
    print('Test: {}/{} (acc: {:.4f}), loss: {:.4f}'.format(running_correct, running_count, epoch_acc, epoch_loss))
    
    return epoch_loss, epoch_acc   


def sample(model, startchar='A'):
    model.eval()
    in_line = lineToTensor(startchar)
    out_line = startchar
    hidden = model.initHidden()
    
    for i in range(100):
        #import pdb; pdb.set_trace()
        out, hidden = model(in_line, hidden)
        c = torch.distributions.categorical.Categorical(logits=out)
        pred = c.sample()
        pred = pred.item()
        if pred == n_letters-1:  # EOS
            break
        else:
            letter = all_letters[pred]
            out_line += letter
            in_line = lineToTensor(letter)
    return out_line

def multiSample(model, n=10, temperature=0.5):
    model.eval()
    model.changeTemperature(temperature)
    startChars = "ABCDEFGHIJKLMNOPRSTUVWZ"
    n_chars = len(startChars)
    lines = []
    for i in range(n):
        start = startChars[np.random.randint(n_chars)]
        line = sample(model,start)
        lines.append(line)
    
    model.changeTemperature(1) #reset back to original
    return lines

num_layers = 2
hidden_size = 100
learning_rate = 1
n_iters = 500
print_every = 1
#plot_every = 500
train_losses = []
test_losses = []

#split categories into 80% 20%, shuffled earlier.
train_categories = all_categories[0:int(0.8*n_categories)] 
test_categories = all_categories[int(0.8*n_categories):]

model = LSTM(n_letters, hidden_size, num_layers)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,10,0.2)

start = time.time()

with open('startrek_gen.txt','w') as f:
    for iter in range(1, n_iters + 1):
        print('Epoch {}, {}'.format(str(iter), timeSince(start)))
        train_loss, train_acc = train(model, train_categories,criterion,optimizer)
        test_loss, test_acc = test(model,test_categories,criterion)
        
        #total_loss += loss
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        if iter % print_every == 0:
            f.write('Epoch {}:\n'.format(iter))
            f.write('Train acc: {}\nTest acc: {}'.format(train_acc,test_acc))
            f.write('Train loss: {}\nTest loss: {}'.format(train_loss,test_loss))
            f.write('\n'.join(multiSample(model))+'\n\n')
            
            #print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            print(multiSample(model))

torch.save(model.state_dict(),'model')

import matplotlib.pyplot as plt

#print(train_losses)
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
#plt.show()
plt.legend()
plt.savefig('Losses.png')