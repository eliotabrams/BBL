# Eliot Abrams
# Food truck location choice

# Code currently runs the BBL commands on a dummy dataset
# The BBL commands are built to be robust to working with the real data.
# I've just barely started working with the real data.

# One approach would be to have this file contain all the food truck specific stuff
# And have the BBL file contain all the code that is generic to the BBL implementation

# Reset the IPython environment
%reset

# Packages
import os as os

# Run BBL script to setup environment and declare functions
%run '/Users/eliotabrams/Desktop/BBL/BBL.py'

# Set seed
np.random.seed(1234)

# Import location data (only observe when a truck parks at a popular location)
#location_data = pd.read_csv('/Users/eliotabrams/Desktop/Data/output.csv', index_col = 0)
location_data = pd.read_csv(
    '/Users/eliotabrams/Desktop/foodtrucklocationdata.csv')

# Print list of food trucks so that I can add type
#os.chdir('/Users/eliotabrams/Desktop/Data')
#location_data.groupby('Truck')['Truck'].count().to_csv('truck_types.csv')

# View some summary statistics on the location data
location_data.sample(10)
location_data.groupby('Truck')['Truck'].count()
location_data.groupby('Date')['Location'].count()
location_data.groupby('Date')['Location'].nunique()
location_data.groupby('Location')['Date'].count()
location_data.groupby(['Truck', 'Date'])['Location'].count()

# Take the first observation for each truck for each day
location_data = location_data.drop_duplicates(['Truck', 'Date'])

# Drop rare trucks, drop non-city locations, drop rare locations (parkings
# here will then later get lumped into an other category)


# Import the truck type data (need to construct for the real dataset somehow)
truck_types = pd.read_csv('/Users/eliotabrams/Desktop/trucktypes.csv')

# Create variables
(cleaned_location_data, state_variables) = make_states(
    location_data=location_data, making_probabilities=True, truck_types=truck_types)

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
                periods=4, discount=.99, state_variables=state_variables,
                N=4, truck_types=truck_types, num_draws=5)

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


results.describe()

# Plot (I'm using pylab within IPython)
pylab
pd.options.display.mpl_style = 'default'
results.plot(subplots=True, layout=(3, 3), kind='hist')
