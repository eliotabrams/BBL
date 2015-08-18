#!/usr/bin/env python

"""
simulate.py: Imports the BBL module and uses the commands therein 
to run a BBL simulation on location data of Chicago Food Trucks
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
##         Simulate         ##
##############################

# Read in data
print g.g

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
    g_bootstrap_sample = g.iloc[np.random.randint(0, len(g), size=50)]
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
