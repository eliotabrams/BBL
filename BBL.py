# Eliot Abrams
# Food truck location choice

# Code sets up the BBL environment

# Packages
import pandas as pd
import numpy as np
import sympy as sp
import datetime as dt
import scipy.optimize as opt

# Constants
COUNT_OF_EMPTY_STATES_REACHED = 0
HIGH_COUNT = 0
HIGH_UNIQUE = 0
HIGH_FREQ = 0

# Sympy Variables (WOULD LIKE TO FIND A CLEANER WAY OF DECLARING AND
# STORING THESE) These are not all used yet, but will come in handy when I 
# update the profit function to include interactions

# Intercept
intercept = sp.Symbol('intercept')

# Days
monday = sp.Symbol('monday')
tuesday = sp.Symbol('tuesday')
wednesday = sp.Symbol('wednesday')
thursday = sp.Symbol('thursday')
friday = sp.Symbol('friday')
saturday = sp.Symbol('saturday')
sunday = sp.Symbol('sunday')
days = [monday, tuesday, wednesday, thursday, friday, saturday, sunday]

# Quarters
q1 = sp.Symbol('q1')
q2 = sp.Symbol('q2')
q3 = sp.Symbol('q3')
q4 = sp.Symbol('q4')
quarters = [q1, q2, q3, q4]

# Locations
CityfrontPlaza = sp.Symbol('CityfrontPlaza')
ClarkandMonroe = sp.Symbol('ClarkandMonroe')
LasalleandAdams = sp.Symbol('LasalleandAdams')
MadisonandWacker = sp.Symbol('MadisonandWacker')
RandolphandColumbus = sp.Symbol('RandolphandColumbus')
UniversityofChicago = sp.Symbol('UniversityofChicago')
WackerandAdams = sp.Symbol('WackerandAdams')
WestChicagoAvenue = sp.Symbol('WestChicagoAvenue')

locations = pd.DataFrame(
    [CityfrontPlaza, ClarkandMonroe, LasalleandAdams, MadisonandWacker, 
     RandolphandColumbus, UniversityofChicago, WackerandAdams, 
     WestChicagoAvenue]).transpose()
locations.columns = ['CityfrontPlaza', 'ClarkandMonroe', 'LasalleandAdams', 
                     'MadisonandWacker', 'RandolphandColumbus',
                     'UniversityofChicago', 'WackerandAdams',
                     'WestChicagoAvenue']

# Main variables
high_historic_count = sp.Symbol('high_historic_count')
high_historic_diversity = sp.Symbol('high_historic_diversity')
high_historic_freq = sp.Symbol('high_historic_freq')
high_current_count = sp.Symbol('high_current_count')
high_current_diversity = sp.Symbol('high_current_diversity')

# Create the quantiles for discretizing
# These should be redone so that they use the same rolling window as the make states below
def set_discretize(location_data, truck_types):
    """
    Create the quantiles for discretizing

    """

    # Tell function which variables are globals
    global HIGH_COUNT
    global HIGH_UNIQUE
    global HIGH_FREQ

    # Find the frequency with which each truck parks at each location_data
    location_data = pd.merge(location_data, truck_types, on='Truck')
    location_data['Date'] = pd.to_datetime(location_data['Date'])
    location_data['Week'] = location_data.Date.dt.week
    parkings_per_location = location_data.groupby(['Location', 'Week']).Type.agg(['count', 'nunique'])
    HIGH_COUNT = parkings_per_location.quantile(q=0.8)[0]*2
    HIGH_UNIQUE = parkings_per_location.quantile(q=0.8)[1]

    # Find the frequency with which each truck parks at each location_data
    truck_parkings_per_location = location_data.groupby(['Truck', 'Location']).Date.count().reset_index()
    truck_parkings = location_data.groupby(['Truck']).Date.count().reset_index()
    truck_parkings_per_location = truck_parkings_per_location.merge(truck_parkings, on='Truck') 
    truck_parkings_per_location['Freq'] = truck_parkings_per_location.Date_x / \
        truck_parkings_per_location.Date_y.apply(float) 
    HIGH_FREQ = truck_parkings_per_location.quantile(q=0.8, axis=0)[2]*10

    return


# Create states as a tuple and add as a column to the input location data 
# Returns where each truck parked at each state
def make_states(actions, making_probabilities, truck_types):
    """
    Takes DataFrame with Truck, Location, and Date and returns DataFrame with created states and also state variables
    """

    # Merge on truck types
    actions = pd.merge(actions, truck_types, on='Truck')

    # Create date range
    actions['Date'] = pd.to_datetime(actions['Date'])
    dates = pd.date_range(start=actions.Date[0], end=actions.Date[len(actions)-1], freq='D')
    dates = dates[dates.dayofweek<5]

    states = pd.DataFrame()
    # Loop over lag 10 days
    for x in xrange(len(dates) - 10):
        location_data = actions[(actions.Date >= dates[x]) & (actions.Date < dates[x+10])]

        # Find the number and diversity of trucks at each location in each week
        grouped_by_year_plus_week_location = location_data.groupby('Location')
        joint_state_variables = grouped_by_year_plus_week_location.Type.agg(['count', 'nunique']).reset_index(
            ['Location']).rename(columns={'count': 'Count', 'nunique': 'Num_Unique'})
        joint_state_variables = joint_state_variables[joint_state_variables.Location!='Other']

        container_table = truck_types.drop('Type', axis=1)
        temp = pd.DataFrame(list(locations.columns), columns=['Location'])

        joint_state_variables = pd.merge(joint_state_variables, temp, how='right', on='Location').fillna(0)
        joint_state_variables['Date'] = dates[x+10]

        # Discretize the values (turn into dummy variables for now).
        joint_state_variables.Count = joint_state_variables.Count.apply(lambda row: int(row >= HIGH_COUNT))
        joint_state_variables.Num_Unique = joint_state_variables.Num_Unique.apply(lambda row: int(row >= HIGH_UNIQUE))

        # Form the pivot table
        joint_state_variables = pd.pivot_table(joint_state_variables,
                                               values=['Count', 'Num_Unique'],
                                               index='Date',
                                               columns='Location').fillna(0).reset_index(['Year_Plus_Week', 'Count', 'Num_Unique'])

        # Collapse the multiple indices
        joint_state_variables.columns = pd.Index(
            [e[0] + e[1] for e in joint_state_variables.columns.tolist()])

        # Find the frequency with which each truck parks at each location_data
        truck_specific_state_variables = location_data.groupby(['Truck', 'Location']).Date.count(
        ).reset_index(['Truck', 'Location']).rename(columns={'Date': 'Truck_Weekly_Frequency'})
        truck_specific_state_variables = truck_specific_state_variables[truck_specific_state_variables.Location!='Other']
        truck_specific_state_variables['Date'] = dates[x+10]

        # Discretize the values (turn into dummy variables for now).
        truck_specific_state_variables.Truck_Weekly_Frequency = truck_specific_state_variables.Truck_Weekly_Frequency.apply(
            lambda row: int(row > HIGH_FREQ))

        # Create container table table (to ensure that all truck location combinations are present)
        container_table = truck_types.drop('Type', axis=1)
        temp = pd.DataFrame(list(locations.columns), columns=['Location'])
        container_table['key'] = 1
        temp['key'] = 1
        container_table = pd.merge(
            container_table, temp, on='key').ix[:, ('Truck', 'Location')]

        # Finish the pivot table
        truck_specific_state_variables = truck_specific_state_variables.append(container_table).fillna(0)
        historic_truck_frequencies = pd.pivot_table(truck_specific_state_variables,
                                                    values='Truck_Weekly_Frequency',
                                                    index='Date',
                                                    columns=['Location', 'Truck']).fillna(0).reset_index()

        # Collapse the multiple indices
        historic_truck_frequencies.columns = pd.Index(
            [e[0] + str(e[1]) for e in historic_truck_frequencies.columns.tolist()])

        # Should be a single entry
        new_vars = pd.merge(joint_state_variables, historic_truck_frequencies, on=['Date'])
        
        # Create quarter
        new_vars['Quarter'] = new_vars.Date.dt.quarter

        # Concatenate the created variables into a single state variable
        new_vars = new_vars.reindex_axis(sorted(new_vars.columns), axis=1)
        state_variables = new_vars.columns
        state_variables = state_variables.drop('Date')
        state_variables = list(state_variables)

        # Add new state to list along with its corresponding date
        states = states.append(
            pd.DataFrame([dates[x+10], tuple(new_vars[state_variables].values[0])]).transpose())

    states.columns = ['Date', 'State']
    return (states, state_variables)


# Calculate P(a_{it} | s_t) (WILL NEED TO REDO WITH A SEIVE LOGIT)
def find_probabilities(actions_with_states):
    """
    Takes DataFrame with Truck, Location, Date, and State and returns DataFrame with action probabilities
    """

    # Find the number of times that each truck takes each action for each state
    numerator = cleaned_location_data.groupby(
        ['Truck', 'Location', 'State'])['Date'].count().reset_index()

    # Find the number of times that each state occurs
    denominator = cleaned_location_data.groupby(
        ['Truck', 'State'])['Date'].count().reset_index()

    # Calculate the probabilities
    probabilities = pd.merge(numerator, denominator, on=['Truck', 'State'])
    probabilities['Probability'] = probabilities.Date_x.apply(float) / probabilities.Date_y.apply(float)
    probabilities = probabilities.drop(['Date_x', 'Date_y'], 1)

    return probabilities


# Find vector of optimal action from probability list and state
def optimal_action(probability_list, state, truck_types):
    """
    Find optimal actions for the trucks at the given state from the Probability DataFrame
    """

    #probability_list = probabilities
    #state = starting_state
    #probability_list.loc[probability_list['State'] == state].empty

    # If the state is not present in the historic data then generate random
    # actions for the trucks
    if probability_list.loc[probability_list['State'] == state].empty:
        action_profile = generate_random_actions(truck_types)

        global COUNT_OF_EMPTY_STATES_REACHED
        COUNT_OF_EMPTY_STATES_REACHED += 1

    # If the state is present, find the optimal action using the Hotz-Miller
    # inversion
    else:
        comparison = probability_list.loc[probability_list['State'] == state]
        comparison['Shock'] = np.random.gumbel(
            loc=0.0, scale=1.0, size=len(comparison.index))
        comparison['Value'] = np.log(
            comparison['Probability']) + comparison['Shock']
        action_profile = comparison.sort('Value', ascending=False).drop_duplicates(
            'Truck').loc[:, ['Truck', 'Location', 'Shock']]

    return action_profile.sort('Truck')


# Find other actions (for now just enable acting randomly and acting certainly)
def generate_random_actions(truck_types):
    """
    Find random actions for the trucks
    """

    # Create a table with all possible actions for all trucks
    action_profile = truck_types.drop('Type', axis=1)
    temp = pd.DataFrame(list(locations.columns), columns=['Location'])
    action_profile['key'] = 1
    temp['key'] = 1
    comparison = pd.merge(action_profile, temp, on='key').ix[:, ('Truck', 'Location')]

    # Generate a random shock
    comparison['Shock'] = np.random.gumbel(
        loc=0.0, scale=1.0, size=len(comparison.index))

    # Return best action for truck (the economics is that I'm putting a null
    # prior over each action and so the action taken according to the
    # Hotz-Miller inversion is just the action with the highest shock value)
    action_profile = comparison.sort('Shock', ascending=False).drop_duplicates(
        'Truck').loc[:, ['Truck', 'Location', 'Shock']]

    return action_profile


def generate_certain_actions(certain_action, truck_types):
    """
    Returns table with specified action for trucks
    """

    action_profile = truck_types.drop('Type', axis=1)
    action_profile['Location'] = certain_action
    action_profile['Shock'] = np.random.gumbel(
        loc=0.0, scale=1.0, size=len(action_profile.index))

    return action_profile


# Calculate profit given current state and action profile 
# This function is specific to the food truck location application
def get_profit(location, truck, shock, df, current_variables, truck_types):
    """
    Builds the period profit vector for a truck
    """

    # Zero out profit if chosen location is Other
    if (location == 'Other'):
        profit = 0

    # Add intercept, day of week indicator, quarter indicator, and shock. Day
    # of week is fed in as a 0-6, but quarter is fed in as 1-4 hence the
    # indexing adjustment for quarter
    else:
        profit = intercept + quarters[df.Quarter[0] - 1] + shock

        # Add historic count and diversity at chosen location
        count_var = 'Count' + location
        num_unique_var = 'Num_Unique' + location
        profit = profit + df[count_var][0] * high_historic_count + \
            df[num_unique_var][0] * high_historic_diversity

        # Add truck's historic frequency at chosen location
        historic_freq_var = location + str(truck)
        profit = profit + df[historic_freq_var][0] * high_historic_freq

        # Add current location variables
        profit = profit + locations[location][0] + current_variables.Count[location] * \
            high_current_count + \
            current_variables.Num_Unique[location] * high_current_diversity

    return profit


# Builds the current period variables and runs the get_profit() vector for each of the trucks
# this function is currently specific to the food truck location application
# but could be made more general. Also, it would be nice to do the discretizing
# of the current period variables through a call to another function
def create_profit_vector(state_variables, state, actions, truck_types):
    """
    Creates the current period variables and then calls get_profit() for each truck
    """

    # Put into data frame
    df = pd.DataFrame([state]).applymap(int)
    df.columns = state_variables

    # Create variables based on current actions
    actions = pd.merge(actions, truck_types, on='Truck')
    current_variables = actions.groupby(['Location'])['Type'].agg(
        ['count', 'nunique']).rename(columns={'count': 'Count', 'nunique': 'Num_Unique'})

    # Discretize the values (turn into dummy variables for now).
    current_variables.Count = current_variables.Count.apply(lambda row: int(row >= HIGH_COUNT))
    current_variables.Num_Unique = current_variables.Num_Unique.apply(lambda row: int(row >= HIGH_UNIQUE))

    # Create profit vector
    Profit_Vector = actions.drop(['Type'], 1)
    Profit_Vector['Profit'] = Profit_Vector.apply(lambda row: get_profit(
        row['Location'], row['Truck'], row['Shock'], df, current_variables, truck_types), axis=1)

    return Profit_Vector.drop(['Location', 'Shock'], 1)


# Update state
def update_state(state, action_sequence, Date, state_variables, truck_types):
    """
    Take the current state and recent history of actions and return the new state (clearing out the action sequence as necessary)
    """

    # If its the first day of the week, return the new state based on the
    # actions from the previous week and reset the sequence of actions that
    # we're keeping track of
    if pd.DatetimeIndex([Date])[0].dayofweek == 0:
        (Values, Labels) = make_states(location_data=action_sequence,
                                       making_probabilities=False, truck_types=truck_types)

        Content = pd.DataFrame([Values.State[0]])
        Content.columns = Labels

        Container = pd.DataFrame(np.zeros(len(state_variables))).transpose()
        Container.columns = state_variables

        new_state = tuple(pd.concat([Container, Content]).fillna(0).iloc[[1]].values[0])

        action_sequence = action_sequence.sort(columns=['Truck', 'Date'], ascending=False)
        action_sequence = action_sequence.drop_duplicates('Truck')

    # Else just update the day of week and quarter (I perfer to keep the state
    # as a tuple generally so that I don't accidently change it)
    else:
        new_state = list(state)
        new_state[state_variables.index('Quarter')] = pd.DatetimeIndex(
            [Date])[0].quarter
        new_state = tuple(new_state)

    return [new_state, action_sequence]


# Simulate a single path
def simulate_single_path(probabilities, starting_state, starting_date,
                         periods, discount, state_variables, truck_id,
                         action_generator, specific_action, truck_types):
    """
    Simulate a single path of actions for a truck and return the value function experienced
    """

    # Set the initial values
    current_date = dt.datetime.strptime(starting_date, '%Y-%m-%d')
    current_state = starting_state
    T = 0
    pdv_profits = np.zeros(len(truck_types))
    action_sequence = pd.DataFrame()

    while T < periods:
        # Find the optimal actions and add to the action sequence
        actions = optimal_action(probabilities, current_state, truck_types)

        # Replace specific truck's action with alternate strategy if requested
        if action_generator == 'Random':
            truck_actions = generate_random_actions(truck_types)
            truck_actions = truck_actions[truck_actions.Truck == truck_id]
            actions = actions[actions.Truck != truck_id]
            actions = actions.append(truck_actions)

        if action_generator == 'Specific':
            truck_actions = generate_certain_actions(specific_action, truck_types)
            truck_actions = truck_actions[truck_actions.Truck == truck_id]
            actions = actions[actions.Truck != truck_id]
            actions = actions.append(truck_actions)

        # Add the date to the current actions
        actions['Date'] = dt.datetime.strftime(current_date, '%Y-%m-%d')

        # Create the profit vector and add to the discounted sum of profits
        period_profits = create_profit_vector(state_variables=state_variables, state=current_state,
                                              actions=actions, truck_types=truck_types)
        pdv_profits += discount ** T * period_profits.Profit

        # Update state (appending current actions to action sequence)
        actions = actions.drop(['Shock'], axis=1)
        action_sequence = action_sequence.append(actions)

        (current_state, action_sequence) = update_state(state=current_state, action_sequence=action_sequence,
                                                        Date=dt.datetime.strftime(
                                                            current_date, '%Y-%m-%d'),
                                                        state_variables=state_variables, truck_types=truck_types)

        # Update counters
        T += 1
        current_date += dt.timedelta(days=1)

    return pd.DataFrame([period_profits.Truck, pdv_profits]).transpose()


# Average over N simulations of the valuation function
# Could combine with the above function if there are efficiency improvements
def find_value_function(probabilities, starting_state, starting_date, periods,
                        discount, state_variables, truck_id, action_generator,
                        specific_action, N, truck_types):
    """
    Average over N simulations of the valuation function
    """

    # Set initial values
    value_functions = np.zeros(len(truck_types))

    # Run N times
    for x in xrange(N):
        Results = simulate_single_path(probabilities=probabilities, starting_state=starting_state,
                                       starting_date=starting_date, periods=periods, discount=discount,
                                       state_variables=state_variables, truck_id=truck_id, action_generator=action_generator,
                                       specific_action=specific_action, truck_types=truck_types)

        value_functions += 1. / N * Results.Profit

    # Format results
    Step_One = pd.DataFrame([Results.Truck, value_functions]).transpose()

    return Step_One[Step_One.Truck == truck_id].Profit.get_values()[0]


# HAVE CHANGED SAMPLE TO HEAD TEMPORARILY TO DEAL WITH OLD VERSION OF PANDAS ON GRID
# Build the terms that go into the objective to the maximization problem
def build_g(probabilities, periods, discount, state_variables, N, truck_types, num_draws):
    """
    Randomly choose num_draws of inequalities to use and estimate the relevant value functions
    """

    # Create columns with truck
    container_table = truck_types.drop('Type', axis=1)

    # Interact trucks with alternative strategies being considered
    temp1 = pd.DataFrame(['Random'], columns=['action_generator'])
    temp2 = pd.DataFrame(list(locations.columns), columns=['specific_action'])
    temp2['action_generator'] = 'Specific'
    temp2 = temp2.append(temp1)
    container_table['key'] = 1
    temp2['key'] = 1
    container_table = pd.merge(container_table, temp2, on='key').ix[
        :, ('Truck', 'action_generator', 'specific_action')]
    container_table = container_table.fillna('')

    # Interact trucks and alternative strategies with all possible starting states
    # being given positive weight
    container_table['key'] = 1
    states = pd.DataFrame(probabilities[probabilities.Truck == 'Yum Dum Truck'].State)
    states['key'] = 1
    container_table = pd.merge(container_table, states, on='key').drop('key', axis=1)

    # Randomly draw requested number of inequalities
    # Cannot use the .sample() method because only have Pandas 0.15.2 on grid
    container_table = container_table.reindex(np.random.permutation(container_table.index))
    container_table = container_table.head(num_draws)

    # Create starting date appropriate for quarter
    # by randomly drawing from the possibilities
    # Currently NOT WORKING! Has everything starting on first day of quarter.
    dates = pd.date_range(start='1/1/2011', end='12/31/2011', freq='D')
    container_table['starting_date'] = container_table.State.apply(
                                    lambda row: pd.DataFrame(
                                        dates[(dates.quarter == row[state_variables.index('Quarter')] 
                                        )]).head(1)[0].apply(str).values[0][:10])

    # Estimate the value functions
    container_table['Value_Function_For_Other_Actions'] = container_table.apply(lambda row:
                    find_value_function(probabilities=probabilities, starting_state=row['State'],
                                        starting_date=row['starting_date'], periods=periods, discount=discount,
                                        state_variables=state_variables, truck_id=row['Truck'],
                                        action_generator=row['action_generator'],
                                        specific_action=row['specific_action'],
                                        N=N, truck_types=truck_types), axis=1)


    container_table['Value_Function'] = container_table.apply(lambda row:
                    find_value_function(probabilities=probabilities, starting_state=row['State'],
                                        starting_date=row['starting_date'], periods=periods, discount=discount,
                                        state_variables=state_variables, truck_id=row['Truck'],
                                        action_generator='Optimal',
                                        specific_action='',
                                        N=N, truck_types=truck_types), axis=1)

    # Form the relevant differences
    container_table['g'] = container_table.Value_Function - \
        container_table.Value_Function_For_Other_Actions

    return container_table


# Estimate the parameters by maximizing the objective (later add a
# weighting vector)
def optimize(g):
    """
    Find parameters by optimizing over the given table of estimated inequalities
    """

    # Build the objective
    g['Terms'] = g.apply(lambda row: sp.Min(row.g, 0) ** 2, axis=1)
    objective = g.Terms.sum()
    variables = list(objective.atoms(sp.Symbol))

    # Turn the objective into a function
    def function(values):
        z = zip(variables, values)
        return float(objective.subs(z))

    # Create the intial guess
    initial_guess = np.ones(len(variables))

    # Optimize!
    return [opt.minimize(function, initial_guess, method='nelder-mead'), variables]




    
