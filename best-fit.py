
from config import *
from fitting import *

import numpy as np
import pandas as pd

err_vals = []
predictions = [['Behavioral Data']*len(settings), range(len(settings)), list(data)]

for i in range(len(models)): # models form config.py
    predict(models[i], pars[i], names[i], err_vals, predictions)

df = pd.DataFrame(err_vals, columns = ['Model', 'Parameter', 'Error'])
print df
df.to_csv(out_dir + 'errors.csv', index = False)

predictions = list(np.array(predictions).transpose())
df = pd.DataFrame(predictions, columns = ['Model', 'Condition', 'Percent Coordination'])
print df
df.to_csv(out_dir + 'predictions.csv', index = False)
