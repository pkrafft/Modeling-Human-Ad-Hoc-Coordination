
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *
from plot_tools import *

### qualitative predictions ###

sns.set(style = 'whitegrid')
sns.set_context('paper', font_scale = 1.8)

df = pd.read_csv(out_dir + 'predictions.csv')
df = df[df['Model'] == 'Behavioral Data']
df = expand(df)

fig, ax = plt.subplots()

g = sns.factorplot(x = 'Condition',
                   y = 'Coordinate',
                   col = 'Model',
                   color = '#E69F00',
                   kind = 'bar',
                   data = df)

(g.set_axis_labels("Knowledge Condition", "Percent Coordination").set_xticklabels(["Private", "Secondary", "Tertiary", "Common"]).set_titles("{col_name}").set(ylim=(0, 1.1)))
g.set_xticklabels(rotation=20)

plt.tight_layout()

g; plt.savefig(plot_dir + 'data.pdf')

sns.set_context('paper', font_scale = 2)

df = pd.read_csv(out_dir + 'predictions.csv')
df = df[df['Model'] != 'Behavioral Data']
df = expand(df)

g = sns.factorplot(x = 'Condition',
                   y = 'Coordinate',
                   col = 'Model',
                   col_order = ['Rational p-Belief',
                                'Matched p-Belief',
                                'Iterated Maximizing',
                                'Iterated Matching'],
                   color = '#56B4E9',
                   kind = 'bar',
                   data = df)

(g.set_axis_labels("Knowledge Condition", "Percent Coordination").set_xticklabels(["Private", "Secondary", "Tertiary", "Common"]).set_titles("{col_name}").set(ylim=(0, 1.1)))
g.set_xticklabels(rotation=20)

plt.tight_layout()

g; plt.savefig(plot_dir + 'predictions.pdf')

### quantitative predictions ###

sns.set(style = 'whitegrid')
sns.set_context('paper', font_scale = 2.5)

df = pd.read_csv(out_dir + 'errors.csv')

g = sns.factorplot(y = 'Model',
                   x = 'Error',
                   kind = 'bar',
                   order = ['Rational p-Belief',
                            'Matched p-Belief',
                            'Iterated Maximizing',
                            'Iterated Matching'],
                   data = df, aspect = 2)

(g.set_axis_labels("Mean Squared Error", "Model"))
#g.set_xticklabels(rotation=90)
#
#plt.tight_layout()

g; plt.savefig(plot_dir + 'errors.pdf')


### collaborating with humans ###

fig, ax = plt.subplots()

sns.set(style = 'white')
sns.set_context('paper', font_scale = 2.5)

ax.set_color_cycle(sns.color_palette("colorblind"))

df = pd.read_csv(out_dir + 'thomas-simulations.csv')

for i,m in enumerate(['Pair Heuristic', 'Private Heuristic', 'Cognitive Strategy']):
    sub = df['Model'] == m
    ax.plot(df[sub]['Risk'], df[sub]['Relative Value'], label = m, lw = 5)

plt.xlabel('Risk')
plt.ylabel('Strategy Marginal Value')
legend = ax.legend(loc='lower left')

plt.tight_layout()

plt.savefig(plot_dir + 'thomas-simulations.pdf')
