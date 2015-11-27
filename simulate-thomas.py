
import pandas as pd
import numpy as np

from config import *
from simulation import *

offsets = get_offsets()

results = []

for o in offsets:
    
    payoffs = [1, 0, o, 0]
    
    results += compare(payoffs)

df = pd.DataFrame(results, columns = ['Risk', 'Relative Value', 'Model'])
print df

df.to_csv(out_dir + 'thomas-simulations.csv', index = False)

