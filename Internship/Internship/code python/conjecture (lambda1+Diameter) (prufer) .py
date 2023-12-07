#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import random
import numpy as np
from copy import deepcopy
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
from statistics import mean
#from scores.score_logger import ScoreLogger
#from joblib import Parallel, delayed
import pickle
import time
import math
import matplotlib.pyplot as plt

start_time = time.time()
N = 12
MYN = N - 2

LEARNING_RATE = 0.001
n_sessions =1000 #number of new individuals per generation
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next generation

FIRST_LAYER_NEURONS = 128
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

n_actions = N 
observation_space = 2*MYN
len_game = MYN
state_dim = (observation_space,)


model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(n_actions, activation="softmax"))
model.build((None, observation_space))
model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = LEARNING_RATE))

#model = load_model("my_model")
print(model.summary())

def unimodal(myList):
    i = 1
    while (i<len(myList) and myList[i]-myList[i-1] <= 0.1):
        i += 1
    while (i<len(myList) and myList[i]-myList[i-1] >= -0.1):
        i += 1
    if i==len(myList):
        return True
        
    i = 1
    while (i<len(myList) and myList[i]-myList[i-1] >= -0.1):
        i += 1
    while (i<len(myList) and myList[i]-myList[i-1] <= 0.1):
        i += 1
    if i==len(myList):
        return True
    return False

def calcScore(state):
    
    tree = nx.from_prufer_sequence(state[:MYN])

    distMat = np.zeros([N,N])
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(tree,i)
        for j in range(N):
            distMat[i][j]=lengths[j]

    evals = np.linalg.eigvalsh(nx.adjacency_matrix(tree).todense())
    evalsRealAbs = np.zeros_like(evals)
    for i in range(len(evals)):
    	evalsRealAbs[i] = abs(evals[i])
    lambda1 = np.max(evalsRealAbs)
	
    diam = np.amax(distMat)
    sumLengths = np.zeros(N,dtype=np.int8)
    sumLengths = np.sum(distMat,axis=0)
    proximity = np.amin(sumLengths)/(N-1.0)


	#Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
	#We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
    myScore = lambda1 + diam - (N-1+2*math.cos(proximity/(N+1)))
    
    if (myScore > 0.00001) :
        print("temps d'exécution : " + str(time.time() - start_time))
        print("lambda1: " + str(lambda1) + "\n proximity: " + str(proximity) + "\n diameter: " + str(diam) + "\n droite: " + str(math.cos(proximity/(N+1))) + "\n myScore: " + str(myScore))
        nx.draw_kamada_kawai(tree)
        plt.show()
        print("\n")
        exit()
    
    return myScore

    
    




                        

def generate_session(agent, n_sessions, verbose = 0):
    """
    Play n_session games using agent neural network.
    Terminate when games finish 
    """
    states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions = np.zeros([n_sessions, len_game], dtype = int)
    state_next = np.zeros([n_sessions,observation_space], dtype = int)
    prob = np.zeros(n_sessions)
    states[:,MYN,0] = 1
    step = 0
    total_score = np.zeros([n_sessions])
    recordsess_time = 0
    play_time = 0
    scorecalc_time = 0
    pred_time = 0
    while (True):
        step += 1        
        tic = time.time()
        prob = agent.predict(states[:,:,step-1], batch_size = n_sessions) 
        toc = time.time()
        pred_time += toc-tic
        
        for i in range(n_sessions):
            tic = time.time()
            myRand = np.random.rand()
            mySum = 0
            for j in range(n_actions):
                mySum += prob[i][j]
                if myRand < mySum:
                    action = j
                    break
            
            actions[i][step-1] = action
            
            state_next[i] = states[i,:,step-1]
            
            if (action > 0):
                state_next[i][step-1] = action
            state_next[i][MYN + step-1] = 0
            if (step < MYN):
                state_next[i][MYN + step] = 1            
            #calculate final score
            terminal = step == MYN
            toc = time.time()
            play_time += toc-tic
            tic = time.time()
            if terminal:
                total_score[i] = calcScore(state_next[i])
            toc = time.time()
            scorecalc_time += toc-tic
            tic = time.time()
        
            # record sessions 
            if not terminal:
                states[i,:,step] = state_next[i]
            
            toc = time.time()
            recordsess_time += toc-tic
            
        
        if terminal:
            break
    #print(time_counter)
    if (verbose):
        print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
    return states, actions, total_score



def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,percentile)

    elite_states = []
    elite_actions = []
    elite_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:        
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                for item in states_batch[i]:
                    elite_states.append(item.tolist())
                for item in actions_batch[i]:
                    elite_actions.append(item)            
            counter -= 1
    elite_states = np.array(elite_states, dtype = int)    
    elite_actions = np.array(elite_actions, dtype = int)    
    return elite_states, elite_actions
    
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    Select all the sessions that will survive to the next generation
    Same as prev essentially, I should combine these
    """

    #np.append is very slow when used in a loop
    #quicker to first convert to a list and then back?
    #Should really use numba here.
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                counter -= 1
    super_states = np.array(super_states, dtype = int)
    super_actions = np.array(super_actions, dtype = int)
    super_rewards = np.array(super_rewards)
    return super_states, super_actions, super_rewards
    

def playgames_given_actions(saved_actions):
    n_sessions = len(saved_actions)
    states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
    state_next = np.zeros([n_sessions,observation_space], dtype = int)
    states[:,MYN,0] = 1
    step = 0
    total_score = np.zeros([n_sessions])
    while (True):
        step += 1                
        for i in range(n_sessions):            
            action = saved_actions[i][step - 1]
            state_next[i] = states[i,:,step-1]
            state_next[i][step-1] = action
            state_next[i][MYN + step-1] = 0
            if (step < MYN):
                state_next[i][MYN + step] = 1            
            terminal = step == MYN            
            if terminal:
                total_score[i] = calcScore(state_next[i])
            if not terminal:
                states[i,:,step] = state_next[i]
        if terminal:
            break
    return states, total_score


super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
#score_logger = ScoreLogger("Myers CEM")
sessgen_time = 0
random_stuff_time = 0
fit_time = 0
score_time = 0
saved_actions = []
"""with open ('best_species_txt_470.txt', 'rb') as fp:
    saved_actions = pickle.load(fp)
saved_actions = []"""



myRand = random.randint(0,1000)

for i in range(50000): #1000 generations should be plenty
    # generate new sessions
    #sessions = Parallel(n_jobs=4)(delayed(generate_session)(model) for j in range(n_sessions))  #"not picklable" error? need to figure this out
    tic = time.time()
    #sessions = Parallel(n_jobs=2)(delayed(generate_session)(model) for j in range(2)) 
    sessions = generate_session(model,n_sessions,1) 
    toc = time.time()
    sessgen_time = toc-tic
    tic = time.time()
    
    
    states_batch = np.array(sessions[0], dtype = int)
    actions_batch = np.array(sessions[1], dtype = int)
    rewards_batch = np.array(sessions[2])
    states_batch = np.transpose(states_batch,axes=[0,2,1])
    #print( actions_batch,rewards_batch)
    
    
    
    #add survivors from previous generation to current population
    #use numba here?
    #states_batch=np.append(states_batch,super_states)
    #print("Example: "+str(actions_batch[0]))
    states_batch = np.append(states_batch,super_states,axis=0)
    #print(states_batch)

    
    actions_batch= actions_batch.tolist()
    for item in super_actions:
        actions_batch.append(item)
    actions_batch = np.array(actions_batch, dtype = int)
    
    rewards_batch= rewards_batch.tolist()
    for item in super_rewards:
        rewards_batch.append(item)
    rewards_batch = np.array(rewards_batch)
    toc = time.time()
    randomshit_time = toc-tic
    tic = time.time()
    #print( actions_batch,rewards_batch)

    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)
    #print("x")
    toc = time.time()
    select1_time = toc-tic
    tic = time.time()
    #print(elite_states,elite_actions)
    super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) 
    
    #print(super_sessions)
    
    toc = time.time()
    select2_time = toc-tic
    tic = time.time()
    
    #We don't want too many individuals to survive
    #Sort the survivors and then pick the first few
    
    super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
    #print(super_sessions)
    super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
    toc = time.time()
    select3_time = toc-tic
    
    
    tic = time.time()
    #print(elite_states,elite_actions)
    #print(elite_actions,to_categorical(elite_actions, num_classes=2) )
    #print(elite_actions,to_categorical(elite_actions, num_classes=n_actions))
    #elite_actions = np.ones(len(elite_actions))
    #elite_actions[1] = 2
    #elite_actions[2] = 0
    model.fit(elite_states, to_categorical(elite_actions, num_classes=n_actions))
    
    toc = time.time()
    fit_time = toc-tic
    tic = time.time()
    
    #if len(super_sessions) > 10:
    #    super_sessions = super_sessions[:10]
        
    super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
    super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
    super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
    if (saved_actions != []):
        saved_sessions = playgames_given_actions(saved_actions)
        saved_states = np.array(saved_sessions[0], dtype = int)
        saved_states = np.transpose(saved_states,axes=[0,2,1])

        saved_rewards = np.array(saved_sessions[1])
        #print(len(super_states[0]))
        super_states = np.append(super_states,saved_states,axis = 0)
        super_actions = np.append(super_actions,np.array(saved_actions),axis = 0)
        saved_actions = []
        super_rewards = np.append(super_rewards,saved_rewards)
    #print("Best: " + str(super_actions[0])  )

    mean_reward = np.mean(super_rewards)    
    #score_logger.add_score(max(mean_reward,-50), i)
    mean_all_reward = np.mean(rewards_batch)    
    toc = time.time()
    score_time = toc-tic
    print("mean reward: " +  str(mean_all_reward) +  "\nSessgen: " + str(sessgen_time) + ", stuff: " + str(random_stuff_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time) + ", number of iterations: " + str(i)) 
    """
    if (i%20 == 1):
        model.save("my_model")
        with open('best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
            pickle.dump(super_actions, fp)
        with open('best_species_txt_'+str(myRand)+'.txt', 'w') as f:
            for item in super_actions:
                f.write(str(item))
                f.write("\n")
        with open('best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
            for item in super_rewards:
                f.write(str(item))
                f.write("\n")
        
    """         

