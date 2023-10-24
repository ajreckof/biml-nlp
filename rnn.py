import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        
        self.i2e = nn.Linear(input_size, embed_size, dtype=torch.double)

        self.i2o = nn.Sequential(
            nn.Linear(embed_size + hidden_size, output_size, dtype=torch.double),
            nn.Softmax(dim = 1)
            )
        self.i2h = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size, dtype=torch.double),
            )

    def forward(self,input):

        # Get the number of columns
        batch_size, num_columns = input.shape
        
        hidden = self.initHidden(batch_size)
        rez = None
        # Iterate over columns
        for i in range(num_columns):
            batch_word = input[:, i]
            rez, hidden = self.forward_word(batch_word, hidden)
        return rez

    def forward_word(self, input, hidden):

        input = F.one_hot(input, num_classes=self.input_size).to(torch.double)

        input = self.i2e(input)
        combined = torch.cat((input, hidden), 1)
        output = self.i2o(combined)
        hidden = self.i2h(combined)
        
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
    