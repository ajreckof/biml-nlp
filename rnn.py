import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        
        self.i2e = nn.Linear(input_size, embed_size)

        self.i2o = nn.Sequential({
            nn.Linear(input_size + embed_size + hidden_size, output_size),
            nn.ReLU()
            })
        self.i2h = nn.Sequential({
            nn.Linear(input_size + embed_size + hidden_size, hidden_size),
            nn.ReLU()
            })
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = self.i2e(input)
        
        combined = torch.cat((input, hidden), 1)
        
        output = self.i2o(combined)
        output = self.softmax(output)
    
        hidden = self.i2h(combined)
        
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)