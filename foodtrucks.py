# Eliot Abrams
# Food truck location choice

# Reset the iPython environment
    %reset

# Run BBL script to setup environment and declare functions
    import os as os
    os.chdir('/Users/eliotabrams/Desktop/BBL')
    %run BBL.py

# Set seed
    np.random.seed(1234)

# Import and clean data location data and import the table of truck types

    # Import location data
        location_data = pd.read_csv(
            '/Users/eliotabrams/Desktop/foodtrucklocationdata.csv')

    # View some summary statistics on the location data
        location_data.sample(10)
        location_data.groupby('Truck')['Truck'].count()
        location_data.groupby('Date')['Location'].count()
        location_data.groupby('Date')['Location'].nunique()
        location_data.groupby('Location')['Date'].count()
        location_data.groupby(['Truck', 'Date'])['Location'].count()

    # Take the first observation for each truck for each day
        location_data = location_data.drop_duplicates(['Truck', 'Date'])

    # Import the truck type data
        truck_types = pd.read_csv('/Users/eliotabrams/Desktop/trucktypes.csv')

# Create variables
    (cleaned_location_data, state_variables) = make_states(
        location_data=location_data, making_probabilities=True, truck_types=truck_types)

# Estimate the coefficients and their standard errors
    results = pd.DataFrame()
    for x in xrange(10):

        # Run stage one (i.e. perform simulation)!
        g = build_g(probabilities=find_probabilities(cleaned_location_data=cleaned_location_data),
                    starting_date='2015-08-10', periods=5, discount=.99, 
                    state_variables=state_variables, N=5, truck_types=truck_types, num_draws=5)

        # Run stage two (i.e. optimize)!
        (res, variables) = optimize(g)

        # Create DataFrame with results
        coefs = list(res.x)
        coefs.append(res.success)
        results = results.append(pd.DataFrame(coefs).transpose())

# Examine results
    variables.append('Converged')
    results.columns = variables
    results = results.reset_index().drop(['index', 'Converged'], axis=1)
    results = results.applymap(float)
    results.describe()

# Plot (I'm using pylab within IPython)
    # Setup space
    pylab
    pd.options.display.mpl_style = 'default'

    # Make histograms
    results.plot(subplots=True, layout=(3, 3), kind='hist')
