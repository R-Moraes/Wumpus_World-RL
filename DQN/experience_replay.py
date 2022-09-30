import numpy as np

class ExpReplay():
    def __init__(self, e_max=50000, e_min=100):
        self._max = e_max   # max number of experiences in the memory
        self._min = e_min   # min number of experiences for training
        self.exp = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []} # total experiences the Agent stored
    
    def get_max(self):
        '''return the max number of experiences in the memory'''
        return self._max
    
    def get_min(self):
        '''return the min number of experiences for training'''
        return self._min
    
    def get_num(self):
        '''return the current number of experiences'''
        return len(self.exp['state'])
    
    def get_batch(self, batch_size=64):
        '''random choose a batch of the experiences for training'''
        idx = np.random.choice(self.get_num(), batch_size, replace=False)
        state = np.array([self.exp['state'][i] for i in idx])
        action = [self.exp['action'][i] for i in idx]
        reward = [self.exp['reward'][i] for i in idx]
        next_state = np.array([self.exp['next_state'][i] for i in idx])
        done = [self.exp['done'][i] for i in idx]

        return state, action, reward, next_state, done

    def add(self, state, action, reward, next_state, done):
        '''remove the oldest experience if the memory is full'''
        if self.get_num() > self.get_max():
            del self.exp['state'][0]
            del self.exp['action'][0]
            del self.exp['reward'][0]
            del self.exp['next_state'][0]
            del self.exp['done'][0]
        
        '''add single experience'''
        self.exp['state'].append(state)
        self.exp['action'].append(action)
        self.exp['reward'].append(reward)
        self.exp['next_state'].append(next_state)
        self.exp['done'].append(done)
