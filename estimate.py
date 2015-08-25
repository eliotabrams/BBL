#!/usr/bin/env python

"""
estimate.py: Uses the g(E_{ia}) terms created by simulate.py to 
estimate the parameters of the value function.
"""

__author__ = 'Eliot Abrams'
__copyright__ = "Copyright (C) 2015 Eliot Abrams"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "eabrams@uchicago.edu"
__status__ = "Production"


"""
# Run BBL script to setup environment and declare functions
import os
os.chdir('/Users/eliotabrams/Desktop/BBL')
"""

# Import packages and force re-creation of module
import BBL
reload(BBL)
from BBL import *

# Set seed
np.random.seed(1234)



##############################
##         Estimate         ##
##############################

# Import the data
g = pd.DataFrame()
for x in xrange(8):
    lines = pd.read_csv('s' + str(x+1)+'.csv', skip_blank_lines=True, header=None)
    g = g.append(lines)
g.columns = ['g']
g = g.reset_index().drop('index', axis=1)
g.g = g.g.apply(lambda row: sp.sympify(row))

# Store
g.to_csv('g.csv')

# Get the point estimate
(res, variables) = optimize(g)
coefs = list(res.x)
coefs.append(res.success)
results = pd.DataFrame(coefs).transpose()
variables.append('Converged')
results.columns = variables
print results.transpose()

# Bootstrap the SEs
bootstrap_results = pd.DataFrame()
for x in xrange(10):
    g_bootstrap_sample = g.iloc[np.random.randint(0, len(g), size=40)]
    (res, variables) = optimize(g_bootstrap_sample)
    coefs = list(res.x)
    coefs.append(res.success)
    bootstrap_results = bootstrap_results.append(pd.DataFrame(coefs).transpose())
variables.append('Converged')
bootstrap_results.columns = variables

# Examine results
bootstrap_results = bootstrap_results.reset_index().drop(['index'], axis=1)
bootstrap_results = bootstrap_results.applymap(float)
print bootstrap_results.describe().transpose()
print bootstrap_results.describe().transpose().sort()[['mean', 'std']]
print bootstrap_results.describe().transpose().sort()[['mean', 'std']].to_latex()

##############################
##        Visualize         ##
##############################
"""
# Plot (I'm using pylab within IPython)
pylab
pd.options.display.mpl_style = 'default'
results.plot(subplots=True, layout=(3, 3), kind='hist')
"""
