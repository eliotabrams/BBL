# Eliot Abrams
# Food truck location choice

# Code currently runs the BBL commands on a dummy dataset
# The BBL commands are built to be robust to working with the real data.
# I've just barely started working with the real data.

# One approach would be to have this file contain all the food truck specific stuff
# And have the BBL file contain all the code that is generic to the BBL
# implementation

# Run BBL script to setup environment and declare functions
#import os
#os.chdir('/Users/eliotabrams/Desktop/BBL')

from BBL import *

# Set seed
np.random.seed(1234)

# Import location data (only observe when a truck parks at a
# popular location)
location_data = pd.read_csv('locations.csv', index_col=0)
location_data['Year'] = pd.to_datetime(location_data['Date']).dt.year

# Drop weekends
location_data = location_data[
    pd.to_datetime(location_data['Date']).dt.weekday < 5]

# Drop rare trucks
truck_counts = location_data.groupby(['Truck',
                                      'Year'])['Truck'].count().unstack().reset_index()
truck_counts['Parkings'] = truck_counts[2014] + truck_counts[2015]
truck_counts = truck_counts[truck_counts.Parkings > 75].sort('Parkings')
location_data = location_data.merge(pd.DataFrame([truck_counts.Truck]
                                                 ).transpose(), on="Truck")

# Drop old data
location_data = location_data[location_data.Year > 2013]

# Drop non-city (Schaumburg) and rare locations
# Parkings here will then later get lumped into an other category
# Three trucks are dropped because they only ever park outside of the city
truck_counts = location_data.groupby(['Location',
                                      'Year'])['Truck'].count().unstack().reset_index()
truck_counts['Parkings'] = truck_counts[2014] + truck_counts[2015]
truck_counts = truck_counts[truck_counts.Parkings > 500].sort('Parkings')
truck_counts = truck_counts[truck_counts.Location != 'Schaumburg Area']
location_data = location_data.merge(pd.DataFrame([truck_counts.Location]
                                                 ).transpose(), on="Location")

# Clean locations
location_data.Location = location_data.Location.str.replace(' ', '')
location_data.Location = location_data.Location.str.replace('600', '')

# Review changes
#location_data.groupby(['Truck', 'Year'])['Truck'].count().unstack().reset_index()
#location_data.groupby(['Location', 'Year'])['Truck'].count().unstack().reset_index()

######################################
# End with 8 locations and 47 trucks #
######################################

# Print list of food trucks so that I can add type by hand
# os.chdir('/Users/eliotabrams/Desktop/Data')
# location_data.groupby('Truck')['Truck'].count().to_csv('trucktypes.csv')

# View some summary statistics on the location data
# location_data.sample(10)
# location_data.groupby('Date').Location.nunique()
# location_data.groupby('Location').Date.count()

# Take the first observation for each truck for each day
# This will need to be more naunced and account for time
location_data = location_data.drop_duplicates(['Truck', 'Date'])

# Create insightful table!
#location_data.pivot(index='Date', columns='Truck', values='Location').tail(31).to_csv('last_month.csv')

# Import the truck type data (need to construct for the real dataset somehow)
truck_types = pd.read_csv('truck_types.csv')
# truck_types.groupby('Type').count()

# Create variables
(cleaned_location_data, state_variables) = make_states(
    location_data=location_data, making_probabilities=True, truck_types=truck_types)

# Reset truck types so DataFrame only contains trucks present in the
# cleaned data
truck_types = pd.merge(truck_types,
                       pd.DataFrame(cleaned_location_data.drop_duplicates('Truck').Truck), on='Truck')

# probabilities=find_probabilities(cleaned_location_data=cleaned_location_data)
# probabilities.to_csv('state.csv')

# Estimate the coefficients and their standard errors
# Periods controls the number of days the simulation runs for.
# N controls the number of simluated paths that go into creating
# the value function. num_draws controls the number of inequalities
# used to identify the parameters.
# For the laws of large numbers in the math to work, I probably need
# periods = 700+, N = 100+, num_draws = 100+, and the xrange to be 100+.
# But need much more computing power to run.
# Try setting all values to 5 to begin.
results = pd.DataFrame()
for x in xrange(1):

    # Run stage one (i.e. perform simulation)!
    g = build_g(probabilities=find_probabilities(cleaned_location_data=cleaned_location_data),
                periods=50, discount=.99, state_variables=state_variables,
                N=1, truck_types=truck_types, num_draws=100)

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
print results

# Plot (I'm using pylab within IPython)
# pylab
#pd.options.display.mpl_style = 'default'
#results.plot(subplots=True, layout=(3, 3), kind='hist')
