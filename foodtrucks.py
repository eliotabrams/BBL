# Eliot Abrams
# Food truck location choice

\Put the state into a dictionary? Not as simple as I thought

# Reset the iPython environment
    %reset

# Run BBL script to setup environment and declare functions
    import os as os
    os.chdir('/Users/eliotabrams/Desktop/BBL')
    %run BBL.py

# Import and clean data location data and import the table of truck types

    # Import location data
        Location_Data = pd.read_csv('/Users/eliotabrams/Desktop/foodtrucklocationdata.csv')

    # View some summary statistics on the location data
        Location_Data.sample(10)
        Location_Data.groupby('Truck')['Truck'].count()
        Location_Data.groupby('Date')['Location'].count()
        Location_Data.groupby('Date')['Location'].nunique()
        Location_Data.groupby('Location')['Date'].count()
        Location_Data.groupby(['Truck', 'Date'])['Location'].count()

    # Take the first observation for each truck for each day
        Location_Data = Location_Data.drop_duplicates(['Truck','Date'])

    # Import the truck type data
        Truck_Types = pd.read_csv('/Users/eliotabrams/Desktop/trucktypes.csv')

# Create variables
    (Locations, State_Variables) = make_states(location_data=Location_Data, making_probabilities=True, Truck_Types=Truck_Types)

# Run stage one (i.e. perform simulation)!
    g = build_g(Probabilities=find_probabilities(Locations), Starting_State=(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0), Starting_Date='2015-08-10', Periods=10, Discount=.99, State_Variables=State_Variables, N=1, Truck_Types=Truck_Types)

# Run stage two (i.e. optimize)!
    res = optimize(g)


\ Bootstrap for standard errors!
