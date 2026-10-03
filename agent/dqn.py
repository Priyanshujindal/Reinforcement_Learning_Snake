import torch
import torch.nn as nn
class DQN(nn.Module):

    def __init__(self,input_size,output_size):
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,output_size)
        )

    def forward(self,x):
        return self.network(x)

if __name__=="__main__":

    model=DQN(11,3)

    state=torch.tensor(
         [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
         dtype=torch.float32
    )

    q_values=model(state)

    print("Q-values")
    print(q_values)