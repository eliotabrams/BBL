#!/usr/bin/env python

"""
foodtrucks.py: Imports the BBL module and uses the commands therein 
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

# Force re-creation of module (MESSY!)
import BBL
reload(BBL)
from BBL import *

# Set seed
np.random.seed(1234)


##############################
## Clean and report on data ##
##############################

# Import location data (only observe when a truck parks at a
# popular location)
location_data = pd.read_csv('locations.csv', index_col=0)

"""
# Raw summary statistics
print len(location_data)
print len(location_data.groupby('Truck').Truck.count())
print len(location_data.groupby('Location').Truck.count())
"""

# Drop old data
location_data['Year'] = pd.to_datetime(location_data['Date']).dt.year
#location_data.groupby(['Truck', 'Year']).Location.count().unstack().to_csv('identify_old.csv')
location_data = location_data[location_data.Year > 2013]
location_data = location_data.drop('Year', axis=1)

# Drop weekends
location_data = location_data[
    pd.to_datetime(location_data['Date']).dt.weekday < 5]

# Drop rare trucks
truck_counts = location_data.groupby(['Truck']).Truck.count()
truck_counts = truck_counts[truck_counts > 66]
truck_counts = pd.DataFrame([truck_counts]).transpose()
truck_counts.columns = ['Count']
truck_counts.reset_index(level=0, inplace=True)
truck_counts = truck_counts.drop('Count', axis=1)
location_data = location_data.merge(truck_counts, on="Truck")

"""
# Print list of food trucks so that I can add type by hand
os.chdir('/Users/eliotabrams/Desktop/Data')
location_data.groupby('Truck')['Truck'].count().to_csv('truck_types.csv')
"""

# Drop dessert and breakfast only trucks
truck_types = pd.read_csv('truck_types.csv')
truck_types = truck_types[truck_types.Type != 'Dessert']
location_data = location_data.merge(pd.DataFrame([truck_types.Truck]
                                                 ).transpose(), on='Truck')
location_data = location_data[location_data.Truck != 'Eastman Egg']

# Drop non-city parkings
location_data = location_data[location_data.Location != 'Schaumburg Area']
location_data = location_data[location_data.Location !=
                              '1815 South Meyers Road, Oakbrook Terrace, IL']

"""
# Summary statistics
print len(location_data)
print len(location_data.groupby('Truck').Truck.count())
print len(location_data.groupby('Location').Truck.count())
"""

# Drop non-lunch parkings (note that time is converted to 24 hour clock)
location_data['Start_Time'] = location_data.Start_Time.apply(
    lambda x: dt.datetime.strptime(x, '%I:%M %p'))
location_data['End_Time'] = location_data.End_Time.apply(
    lambda x: dt.datetime.strptime(x, '%I:%M %p'))
location_data = location_data[(location_data.Start_Time.dt.hour < 12)
                              & (location_data.End_Time.dt.hour > 12)]
location_data = location_data.drop(['Start_Time', 'End_Time'], axis=1)

# Because of data errors there are still truck, date duplicates
location_data = location_data.drop_duplicates(['Truck', 'Date'])

"""
# Summary statistics
print len(location_data)
print len(location_data.groupby('Truck').Truck.count())
print len(location_data.groupby('Location').Truck.count())
"""

# Drop rare locations and clean remaining locaiton names
# Parkings here will then later get lumped into an other category
# I particularly want to exclude Daley Plaza as it only opens
# for lunch service on one day a week (and the day cycles)
truck_counts = location_data.groupby(['Location']).Truck.count()
truck_counts = truck_counts[truck_counts > 150]
truck_counts = pd.DataFrame([truck_counts]).transpose()
truck_counts.columns = ['Count']
truck_counts.reset_index(level=0, inplace=True)
truck_counts = truck_counts.drop('Count', axis=1)
location_data = location_data.merge(truck_counts, on='Location')

"""
# Summary statistics
print len(location_data)
print len(location_data.groupby('Truck').Truck.count())
print len(location_data.groupby('Location').Truck.count())
temp = location_data.groupby('Location').Truck.count()
temp.sort()
print temp
"""

"""
# Create tables!
table = location_data[(location_data.Truck == "Jack's Fork in the Road")
                      | (location_data.Truck == "La Boulangerie")
                      | (location_data.Truck == "The Fat Shallot")
                      | (location_data.Truck == "The Jibarito Stop")]

table = location_data.pivot(index='Date', columns='Truck', 
                    values='Location').tail(31).to_csv('last_month.csv')

table = location_data.groupby('Truck').Truck.count()
table = pd.DataFrame([table]).transpose()
table.columns = ['Parkings']
table.reset_index(level=0, inplace=True)
table = pd.merge(table, truck_types, on='Truck')
table.to_csv('trucks.csv')
"""

# Clean the location names
location_data.Location = location_data.Location.str.replace(' ', '')
location_data.Location = location_data.Location.str.replace('600', '')
location_data.Location = location_data.Location.str.replace('450N.', '')

# Complete panel if making probabilities (else complete by construction)
location_data = location_data.pivot(
    index='Date', columns='Truck', values='Location')
location_data = location_data.unstack().reset_index(
    name='Location')
location_data.Location = location_data.Location.fillna('Other')

# Reset truck types so DataFrame only contains trucks present in the
# final data
truck_types = pd.merge(truck_types, pd.DataFrame(
    location_data.drop_duplicates('Truck').Truck), on='Truck')


##############################
##      Run Estimation      ##
##############################

# Create states
(locations_w_states, state_variables) = make_states(
    location_data=location_data, making_probabilities=True, truck_types=truck_types)
states = locations_w_states.State.drop_duplicates()

# Create probabilities
probabilities = find_probabilities(
    locations_w_states=locations_w_states, state_variables=state_variables)

"""
# Examine results (note that an other location has been added for a total of 9 locations)
  print BBL.HIGH_COUNT
  print BBL.HIGH_UNIQUE
  print BBL.HIGH_FREQ
  print len(state_variables)
  print len(locations_w_states)
  print len(locations_w_states.groupby('Truck').Truck.count())
  print len(locations_w_states.groupby('Location').Truck.count())

# Get a sense of the state space
# I need the states to repeat over time in order to have any hope that there is 
# information in the state that the truck is using to make a decision
# The full state does not repeat over time
# Thankfully, the sub-state that you'd expect the truck to be paying attention to
# does repeat significantly over time
  print len(locations_w_states.State.value_counts())
  print len(locations_w_states.Date.value_counts())
  truck_types = truck_types.reindex(np.random.permutation(truck_types.index))    
  truck = truck_types.head(1).Truck.values[0]
  test = list(locations.columns + truck) + list('Count' 
    + locations.columns) + list('Num_Unique' + locations.columns) + ['Quarter', 'Day_Of_Week']
  locations_w_states['Sub_State'] = locations_w_states[test].values.tolist()
  locations_w_states.Sub_State = locations_w_states.Sub_State.apply(tuple)
  print len(locations_w_states.Sub_State.value_counts())
  print len(locations_w_states.Date.value_counts())
  temp = locations_w_states.groupby('Sub_State').Date.agg('nunique')
  temp.describe()

# View probabilities
  probabilities = find_probabilities(
    locations_w_states=locations_w_states, state_variables=state_variables)
  pylab
  pd.options.display.mpl_style = 'default'
  probabilities.hist()
"""

# Estimate the coefficients and their standard errors
# Periods controls the number of days the simulation runs for.
# N controls the number of simluated paths that go into creating
# the value function. num_draws controls the number of inequalities
# used to identify the parameters.
# For the laws of large numbers in the math to work, I probably need
# periods = 700+, N = 100+, num_draws = 100+, and the xrange to be 100+.
# But need much more computing power to run.
# Try setting xrange = 1, periods = 10, N=1, and num_draws=20 to begin.
results = pd.DataFrame()
for x in xrange(1):

    # Run stage one (i.e. perform simulation)!
    g = build_g(states=states, probabilities=probabilities, periods=1, discount=.99,
                state_variables=state_variables,
                N=1, truck_types=truck_types, num_draws=1)

    # Run stage two (i.e. optimize)!
    (res, variables) = optimize(g)

    # Create DataFrame with results
    coefs = list(res.x)
    coefs.append(res.success)
    results = results.append(pd.DataFrame(coefs).transpose())

# Examine results
variables.append('Converged')
results.columns = variables
results = results.reset_index().drop(['index'], axis=1)
results = results.applymap(float)
results.to_csv('results.csv')
print results
print results.describe().transpose().sort()[['mean', 'std']].to_latex()

##############################
##        Visualize         ##
##############################
"""
# Plot (I'm using pylab within IPython)
pylab
pd.options.display.mpl_style = 'default'
results.plot(subplots=True, layout=(3, 3), kind='hist')
"""
