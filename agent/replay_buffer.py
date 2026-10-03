from collections import deque
import random

class ReplayBuffer:

    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    
    def push(self,state,action,reward,next_state,done):
        experience=(state,action,reward,next_state,done)

        self.buffer.append(experience)

    def sample(self,batch_size):
        batch=random.sample(self.buffer,batch_size)

        states,actions,rewards,next_states,dones=zip(*batch)

        return states,actions,rewards,next_states,dones
    
    def __len__(self):
        return len(self.buffer)

if __name__=="__main__":

    buffer=ReplayBuffer(5)

    buffer.push(
        [1,2,3],
        0,
        1,
        [4,5,6],
        False
    )

    print("Buffer size",len(buffer))

    states,actions,rewards,next_states,dones=buffer.sample(1)

    print("states:",states)
    print("action:",actions)
    print("reward:",rewards)
    print("next_states:",next_states)
    print("dones:",dones)
