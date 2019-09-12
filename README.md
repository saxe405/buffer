Definitios.py -> the basis definitions of number of tiles, probility distributions
head_movements.py -> all computations related to navigation patterns
transition_rules.py -> creating the probability transition matrix which uses the computations in head_movements.py
channel_rules.py -> model for channel quality changes 
streaming.py -> a model for comparison where one does not look beyond the next segment or take into account the network cost
save_plots -> generate the necessary plots
comparison.py -> compare the approach with streaming and short term caching with this model. This also has an option of path analysis using the 5G traces

cache_mdp.py -> does the value iteration and generates the optimal MDP policy. 
