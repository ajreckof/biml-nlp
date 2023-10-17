import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.hidden = self.initHidden(batch_size)
        
        self.i2e = nn.Linear(input_size, embed_size, dtype=torch.double)

        self.i2o = nn.Sequential(
            nn.Linear(embed_size + hidden_size, output_size, dtype=torch.double),
            nn.ReLU()
            )
        self.i2h = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size, dtype=torch.double),
            nn.ReLU()
            )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        
        input = F.one_hot(input, num_classes=self.input_size)
        input = input.to(torch.double)
        
        input = self.i2e(input)
        
        combined = torch.cat((input, self.hidden), 1)
        
        output = self.i2o(combined)
        output = self.softmax(output)
    
        self.hidden = self.i2h(combined)
        
        return output

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
    
    def resetHidden(self):
        self.hidden = self.initHidden(self.hidden.shape[0])