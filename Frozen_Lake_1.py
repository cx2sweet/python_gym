import numpy as np
import random
import gym
import time
from IPython.display import clear_output

#Source
#https://www.youtube.com/watch?v=QK_PP_2KgGE&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=9&ab_channel=deeplizard


#%%
env=gym.make('FrozenLake-v0')
action_space_size=env.action_space.n
state_space_size=env.observation_space.n

q_table=np.zeros((state_space_size,action_space_size))
print(q_table)


num_episodes=10000
max_steps_per_episode=100

learning_rate=.1
discount_rate=.99

exploration_rate=1
max_exploration_rate=1
min_exploration_rate=.01
exploration_decay_rate=.001
#%%
rewards_all_episodes=[]

#Q-learning
for episode in range(num_episodes):
    state=env.reset()
    done=False
    rewards_current_episode=0
    
    for step in range(max_steps_per_episode):
        exploration_rate_threshold=random.uniform(0,1)
        if exploration_rate<exploration_rate_threshold:
            action=np.argmax(q_table[state,:])
        else:
            action=env.action_space.sample()
            
        new_state, reward, done, info = env.step(action)
        
        #Update Q-table
        q_table[state, action] = q_table[state, action] * (1-learning_rate) + \
            learning_rate*(reward + discount_rate*np.max(q_table[new_state,:]))
            
        state=new_state
        rewards_current_episode+=reward
        
        if done:
            break
        
        #exploration_rate_decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
        
        rewards_all_episodes.append(rewards_current_episode)
        



#%%
for episode in range(3):
    state=env.reset()
    done=False
    print("***Episode ",episode+1, "***\n\n\n\n")
    time.sleep(1)
    
    for step in range(max_steps_per_episode):
        #clear_output
        env.render()
        time.sleep(0.3)
        
        action=np.argmax(q_table[state,:])
        new_state,reward,done,info=env.step(action)
        
        if done:
            env.render()
            if reward==1:
                print("Success!!!")
                time.sleep(3)
            else: 
                print('Nope :(')
                time.sleep(3)
            break
        
        state=new_state

env.close()











