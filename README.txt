
Instructions:
  
set configuration in config.py

run:
sh run.sh

results will be generated in the directory specified in config.py, default: ../results/
output will be generated in the directory specified in config.py, default: ../plots/

-

*** If you use this code, please cite: ***

Peter Krafft, Chris Baker, Alex Pentland, and Joshua
Tenenbaum. (2016). Modeling Human Ad Hoc Coordination. The Thirtieth
AAAI Conference on Artificial Intelligence (AAAI).

----

Descriptions of Files:

-- Main Files --

best-fit.py : script for checking model fit to data

config.py : configuration file

fitting.py : functions for fitting models to data

heuristics.py : alternative models for simulations

info_partition.py : functions for creating information partitions given a generative world model 

levelk.py : level-k iterated reasoning model

model.py : abstract class that specifies model methods

pbelief.py : code for computing common p-belief (see supplementary materials of paper for correctness proof)

player.py : generic functions for playing against a specified opponent, used in level-k and simulations

plot.py : script for plotting results

plot_tools.py : utility functions for plotting results

run.sh : script for generating all plots from paper text

simulate-thomas.py : script for simulating human-agent interaction

simulation.py : supporting functions for simulating human-agent interaction

speaker_model.py : model for loud speaker, described in the paper text

test.sh : script for running unit tests

thomas_model.py : model for primary/secondary/tertiary knowledge conditions, described in the paper text

utils.py : a couple useful general-purpose functions


-- Other Model File for Examples and Testing --

asym_model.py : asymmetric observation model

extended_thomas_model : alternative model of thomas data

simple_model.py : simplest possible observation model
    
----

Copyright (c) 2015 Peter Krafft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

