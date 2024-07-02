import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

dfOrig= pd.read_csv('Data/Synthetic Data.csv')
df = dfOrig.copy()

class RiseEnv(gym.Env):
    def __init__(self,df,req_gr,curr_vals):
        self.df = df    #Filtered df that has temp>=current temp and has range of required gr values
        self.req_gr = req_gr
        """
        Curr_vals is the current state of experiment => A list of temp,gr,volt,pre,time
        State => [700, 4.1, Pre, Vol, Time]   => ?, 4.8
        WE neeed to know the temperature set to achieve required growth rate.
        
        We have a dataframe that has growth rate values within +/-5 of the required growth rate 4.8
        The df has also temperature values that are >= the current temperature

        If an action is taken, we check all the values in the growth rate values from the df for the specific temperature.
        |--- If no such temperatures are present with +- 25, reward will be negative of absolute of (choosen action and current temperature )/10
        |--- Else, we check the distance between the temperature and current temperature, reward will be same as above
            |--- the state will be the growth rate that is of least distance from the required growth rate. Add that distance to the reward.
        The environment is done if it has choosen the least temperature that has required growth rate range from the dataframe.

        """
        self.curr_vals = curr_vals[:]
        # self.index = np.random.randint(len(df))
        self.state = curr_vals[:]
        self.observation_space = spaces.Box(50.0,150.0)
        self.action_space = spaces.Discrete(300)

    
    def step(self,action):
        # dfTemp = self.df[self.df['GrowthRate'].between(0.95*self.req_gr, 1.05*self.req_gr)]
        reward = 0
        done = False
        action+=550
        unique_temps = list(set(self.df['Temperature']))
        print(action)
        if len(self.df)>1 and min(self.df['Temperature']) in list(range(action-(25),action+(26))):
        # if min(self.df['Temperature']) in list(range(action-(25/850),action+(26/850))):
            done = True
            # reward+=abs(self.req_gr-self.state[1])
            reward+=850-action
            # reward+=1-action
            self.state[0] = action
            self.state[1] = min(self.df[self.df['Temperature']==min(self.df['Temperature'])]['GrowthRate'])
            
            return self.state[1],reward,done,{}
        
        

        if action>825:
        # if action>1:
            done = True
            reward-=abs(action-self.state[0])
            return self.state[1],done,reward,{}
        
        
        for i in unique_temps:
            if action in list(range(i-25,i+26)):
            # if action in list(range(i-25/850,i+26/850)):
                self.state[1] = min(self.df[self.df['Temperature'] == i]['GrowthRate']) if len(self.df)>1 else self.state[1]
                self.state[0] = action
                reward-=abs(action-self.state[0])
    
        #We return only self.state[1] i.e., growth rate as observation

        return self.state[1],reward,done,{}
    
    def _get_obs(self,curr_sett):
        return curr_sett[2]

    def reset(self):
        return self.curr_vals[1]

    def render(self):
        pass

    def close(self):
        pass


# curr_vals = [700,4.2,1,1,20]
# req_gr = 6.0
# df = df.groupby('Experiment').apply(lambda x: x.head(len(x)//2))
# print(df[df['Experiment'] == 1])
# print(len(dfOrig))
# dfGR = df[df['GrowthRate'].between(0.95*req_gr, 1.05*req_gr)]
# dfTemp = dfGR[dfGR['Temperature']>=curr_vals[0]]
# env = RiseEnv(dfTemp,req_gr,curr_vals)
# print(dfTemp)
# print(env.observation_space.shape)



#  reward=0
#         #State => Temp,GR,Volt,Pres,Time
        
#         # info = {'Voltage': self.state[3],'Pressure': self.state[4],'Time':self.state[5]}
#         info = {'Time_Takes': self.state[5]}
#         if abs(self.req_gr-self.state[2]) <= 0.2:
#             done = True
#             reward+=abs(self.req_gr-self.state[2])

#         if action < self.state[0] or action >= 850:
#             reward-=5
#             return self.state[1],reward,done,info
        
#         if abs(action-self.state[0]) > 100:
#             reward-=abs(action-self.state[0])/100
        
#         else:
#             # self.df[self.df['Temperature'] ]
#             self.state[0] = action
            
#         if len(self.df[self.df['Temperature'].between(action-25,action+25)])>0:
#             reward+=(self.state[0]-action)    #If it gets closer, give more reward i.e., lower temperature gets more reward
        
        
        
#         observation = self._get_obs(self.state)
#         return observation, reward, done, info