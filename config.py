
import os

from levelk import *
from pbelief import *
from heuristics import *
from thomas_model import *
from speaker_model import *
from player import *
from utils import *

out_dir = '../results/'
plot_dir = '../plots/'

try:
    os.makedirs(out_dir)
    os.makedirs(plot_dir)
except:
    pass

TEST = False # set flag to True and use run.sh to run integration test

### parameters ###

delta = 0.25 # max entropy choice for P(dog day) < 0.5
p = 0.5 # unconditional max entropy choice for message probability 

settings = [(1,1,0,1,0), (1,1,1,0,1), (1,1,1,1,0), (1,1)] # settings described in paper
#settings = [(1,1,0,0,0), (1,1,1,0,1), (1,1,1,1,0), (1,1)] # plausible alternative setting

### values specified in Thomas paper ###

payoffs = [1.1, 0, 1, 0.4] # payoffs from Thomas et al., also [a,b,c,d] in table 1 of our paper

messenger = ThomasModel(delta, p)
speaker = SpeakerModel(delta, p)

players = [0,1,0,0] # which player the human participant plays, specified by the Thomas et al. paper
worlds = [messenger, messenger, messenger, speaker] # settings specified by the Thomas et al. paper

if TEST:
    playA = [25, 50, 50, 100] # total participants who try to coordinate
    playN = [100, 100, 100, 100] # total participants
else:
    #data received from Kyle Thomas 
    playA = [15, 41, 44, 52] # total participants who try to coordinate
    playN = [75, 72, 69, 60] # total participants
data = np.array(playA)/np.array(playN,dtype=float) # human levels of coordination

### models to check ###

models = [
    lambda par,model,player,state: coordinate(model, player, par, state, False, payoffs),
    lambda par,model,player,state: coordinate(model, player, par, state, True, payoffs),
    lambda par,model,player,state: common_p_belief(model,player,state),
    lambda par,model,player,state: int(common_p_belief(model,player,state) > threshold)]

names = ['Iterated Matching', 'Iterated Maximizing', 'Matched p-Belief', 'Rational p-Belief']

k_values = [0, 1, 2, 3, 4] # parameter sweep for iterated (level-k) models
pars = [k_values, k_values, [None], [None]]

threshold = optimal_threshold(payoffs) # optimal threshold for rational p-belief

### models to simulate ###

def player_model(model, player, state):
    return common_p_belief(model, player, state)

simulation_models = [lambda world,player,state,payoffs: alone(world, player, state),
                     lambda world,player,state,payoffs: together(world, player, state),
                     lambda world,player,state,payoffs: play_with(world, player,
                                                          lambda x: player_model(world,1-player, x), state, payoffs, True)]

simulation_names = ['Private Heuristic', 'Pair Heuristic', 'Cognitive Strategy']

