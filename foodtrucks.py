# Eliot Abrams
# Food truck location choice


\ Put the variables and the state in a dictionary!
\ Testing

# Reset the iPython environment
    %reset

# Packages
    import pandas as pd
    import numpy as np
    import sympy as sp
    import datetime as dt

# Global variables (put discount and periods here too!)
        num_trucks = 3
        num_locations = 3
        COUNT_OF_EMPTY_STATES_REACHED = 0
        STATES_REACHED = pd.DataFrame()
        DATES = pd.DataFrame()
        TIMES_CALLED = 0

# Sympy Variables (WOULD LIKE TO FIND A BETTER WAY OF DOING THIS)

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
        locationA = sp.Symbol('locationA')
        locationB = sp.Symbol('locationB')
        locationC = sp.Symbol('locationC')
        locationO = sp.Symbol('locationO')

        locations = pd.DataFrame([locationA, locationB, locationC, locationO]).transpose()
        locations.columns = ['A', 'B', 'C', 'O']

    # Other variables
        high_historic_count = sp.Symbol('high_historic_count')
        high_historic_diversity = sp.Symbol('high_historic_diversity')
        high_historic_freq = sp.Symbol('high_historic_freq')
        high_current_count = sp.Symbol('high_current_count')
        high_current_diversity = sp.Symbol('high_current_diversity')

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

# Functions (MOVE TO A SEPARATE FILE LATER)
    # FUNCTION: Create state as a vector (indicating location) storing an array (holding the variable values). Currently merges in the truck type data... may want to change this
        def make_states(location_data, making_probabilities):
            "Function takes a dataset with Truck, Location, and Date (as a datetime variable)"

            # Complete panel
            if making_probabilities:
                location_data = location_data.pivot(index='Date', columns='Truck', values='Location')
                location_data = location_data.unstack().reset_index(name='Location').fillna('O')

            # Merge on truck types
            location_data = pd.merge(location_data, Truck_Types, on='Truck')

            # Create datetime variables
            location_data['Date'] = pd.to_datetime(location_data['Date'])

            # Create time variables
            location_data['Year_Plus_Week'] = location_data['Date'].dt.week + location_data['Date'].dt.year
            location_data['Day_Of_Week'] = location_data['Date'].dt.dayofweek
            location_data['Quarter'] = location_data['Date'].dt.quarter

            # Find the number and diversity of trucks at each location in each week
            # Form the pivot table
            Grouped_By_Year_Plus_Week_Location = location_data.groupby(['Year_Plus_Week', 'Location'])
            Joint_State_Variables = Grouped_By_Year_Plus_Week_Location['Type'].agg(['count', 'nunique']).reset_index(['Location', 'Year_Plus_Week']).rename(columns={'count':'Count', 'nunique':'Num_Unique'})
            Joint_State_Variables = pd.pivot_table(Joint_State_Variables, values=['Count', 'Num_Unique'], index='Year_Plus_Week',columns='Location').fillna(0).reset_index(['Year_Plus_Week', 'Count', 'Num_Unique'])

            # Collapse the multiple indices
            Joint_State_Variables.columns = pd.Index([e[0] + e[1] for e in Joint_State_Variables.columns.tolist()])

            # Discretize the values (turn into dummy variables for now)
            Joint_State_Variables[Joint_State_Variables.ix[:, Joint_State_Variables.columns != 'Year_Plus_Week'] <= 4] = 0
            Joint_State_Variables[Joint_State_Variables.ix[:, Joint_State_Variables.columns != 'Year_Plus_Week'] > 4] = 1

            # Find the frequency with which each truck parks at each location_data

            # Form the pivot table
            Truck_Specific_State_Variables = location_data.groupby(['Truck','Year_Plus_Week', 'Location'])['Date'].count().reset_index(['Truck', 'Location', 'Year_Plus_Week']).rename(columns = {'Date':'Truck_Weekly_Frequency'})

            # Create container table table
            Container_Table = Truck_Types.drop('Type', axis=1)
            temp = pd.DataFrame(list(location_data.columns), columns=['Location'])
            Container_Table['key'] = 1
            temp['key'] = 1
            Container_Table = pd.merge(Container_Table, temp, on='key').ix[:, ('Truck','Location')]

            Truck_Specific_State_Variables = Truck_Specific_State_Variables.append(Container_Table).fillna(0)

            Historic_Truck_Frequencies = pd.pivot_table(Truck_Specific_State_Variables, values='Truck_Weekly_Frequency', index='Year_Plus_Week', columns=['Location','Truck']).fillna(0).reset_index()

            Historic_Truck_Frequencies = Historic_Truck_Frequencies[Historic_Truck_Frequencies.Year_Plus_Week != 0]


            # Collapse the multiple indices
            Historic_Truck_Frequencies.columns = pd.Index([e[0] + str(e[1]) for e in Historic_Truck_Frequencies.columns.tolist()])

            # Discretize the values (turn into dummy variables for now)
            Historic_Truck_Frequencies[Historic_Truck_Frequencies.ix[:, Historic_Truck_Frequencies.columns != 'Year_Plus_Week'] > 0] = 1
                
            # Merge these new variables onto the location data on with a lag
            if making_probabilities:
                Joint_State_Variables['Year_Plus_Week'] = Joint_State_Variables['Year_Plus_Week'] + 1
                Historic_Truck_Frequencies['Year_Plus_Week'] = Historic_Truck_Frequencies['Year_Plus_Week'] + 1

            # Merge (Note that observations that are not matched are being dropped)
            location_data = pd.merge(location_data, Joint_State_Variables, on=['Year_Plus_Week'])
            location_data = pd.merge(location_data, Historic_Truck_Frequencies, on=['Year_Plus_Week'])

            # Concatenate the created variables into a single state variable
            location_data = location_data.reindex_axis(sorted(location_data.columns), axis=1)

            State_Variables = location_data.columns.tolist()
            State_Variables.remove('Truck')
            State_Variables.remove('Date')
            State_Variables.remove('Location')
            State_Variables.remove('Type')
            State_Variables.remove('Year_Plus_Week')

            location_data['State'] = location_data[State_Variables].values.tolist()
            location_data.State = location_data.State.apply(tuple)

            return [location_data, State_Variables]

    # FUNCTION: Calculate P(a_{it} | s_t) REDO as it LOOKS LIKE I'LL NEED TO DO A SEIVE LOGIT OR SOMETHING ELSE! THE STATE SPACE IS TOO BIG!!!!
        def find_probabilities(Locations):

            # Find the number of times that each truck takes each action for each state
            Numerator = Locations.groupby(['Truck', 'Location', 'State'])['Date'].count().reset_index()

            # Find the number of times that each state occurs
            Denominator = Locations.groupby(['Truck', 'State'])['Date'].count().reset_index()

            # Calculate the probabilities
            Probabilities = pd.merge(Numerator, Denominator, on=['Truck', 'State'])
            Probabilities['Probability'] = Probabilities.Date_x / Probabilities.Date_y.apply(float)
            Probabilities = Probabilities.drop(['Date_x', 'Date_y'], 1)

            return Probabilities

    # FUNCTION: Find vector of optimal action from probability list and state THIS IS TRHOWING UP A WEIRD WARNING!!!
        def optimal_action(Probability_List, State):
            "Find optimal action from probability list, state, and truck id"
            
            # If the state is not present in the historic data then generate random actions for the trucks
            if Probability_List.loc[Probability_List['State'] == State].empty:
                Action_Profile = generate_random_actions()

                global COUNT_OF_EMPTY_STATES_REACHED
                COUNT_OF_EMPTY_STATES_REACHED += 1

            # If the state is present, find the optimal action using the Hotz-Miller inversion
            else:
                Comparison = Probability_List.loc[Probability_List['State'] == State]
                Comparison['Shock'] = np.random.gumbel(loc=0.0, scale=1.0, size=len(Comparison.index))
                Comparison['Value'] = np.log(Comparison['Probability']) + Comparison['Shock']
                Action_Profile = Comparison.sort('Value', ascending=False).drop_duplicates('Truck').loc[:, ['Truck','Location', 'Shock']]

            return Action_Profile.sort('Truck')

    # FUNCTIONs: Find other action (as a function of the state and the strategy or build one for each strategy)
        def generate_random_actions():
            ""

            # Create a table with all possible actions for all trucks
            Action_Profile = Truck_Types.drop('Type', axis=1)
            temp = pd.DataFrame(list(locations.columns), columns=['Location'])
            Action_Profile['key'] = 1
            temp['key'] = 1
            Comparison = pd.merge(Action_Profile, temp, on='key').ix[:, ('Truck','Location')]

            # Generate a random shock
            Comparison['Shock'] = np.random.gumbel(loc=0.0, scale=1.0, size=len(Comparison.index))

            # Return best action for truck (the economics is that I'm putting a null prior over each action and so the action taken according to the Hotz-Miller inversion is just the action with the highest shock value)
            Action_Profile = Comparison.sort('Shock', ascending=False).drop_duplicates('Truck').loc[:, ['Truck','Location', 'Shock']]
            
            return Action_Profile

        def generate_certain_actions(Certain_Action):
            ""

            Action_Profile = Truck_Types.drop('Type', axis=1)
            Action_Profile['Location'] = Certain_Action
            Action_Profile['Shock'] = np.random.gumbel(loc=0.0, scale=1.0, size=len(Action_Profile.index))

            return Action_Profile

    # FUNCTION: Calculate profit given current state and action profile WOULD REALLY LIKE TO GENERALIZE
        def get_profit(location, truck, shock, df, Current_Variables):

            # Add intercept, day of week indicator, quarter indicator, and shock. Day of week is fed in as a 0-6, but quarter is fed in as 1-4 hence the indexing adjustment for quarter
            profit = intercept + days[df.Day_Of_Week[0]] + quarters[df.Quarter[0]-1] + shock

            # Add historic count and diversity at chosen location
            count_var = 'Count' + location
            num_unique_var = 'Num_Unique' + location
            profit = profit + df[count_var][0]*high_historic_count + df[num_unique_var][0]*high_historic_diversity

            # Add truck's historic frequency at chosen location
            historic_freq_var = location + str(truck)
            profit = profit + df[historic_freq_var][0]*high_historic_freq

            # Add current location variables
            profit = profit + locations[location][0] + Current_Variables.Count[location]* high_current_count + Current_Variables.Num_Unique[location]* high_current_diversity

            return profit

        def create_profit_vector(State_Variables, State, Actions):
            "Calculate profit given current state and action profile"

            # Put into data frame
            df = pd.DataFrame([State]).applymap(int)
            df.columns = State_Variables

            # Create variables based on current actions and discretize
            Actions = pd.merge(Actions, Truck_Types, on='Truck')
            Current_Variables = Actions.groupby(['Location'])['Type'].agg(['count', 'nunique']).rename(columns={'count':'Count', 'nunique':'Num_Unique'})
            Current_Variables[Current_Variables <= 4] = 0
            Current_Variables[Current_Variables > 4] = 1

            # Create profit vector
            Profit_Vector = Actions.drop(['Type'], 1)
            Profit_Vector['Profit'] = Profit_Vector.apply(lambda row: get_profit(row['Location'], row['Truck'], row['Shock'], df, Current_Variables), axis=1)

            return Profit_Vector.drop(['Location', 'Shock'], 1)
           
    # FUNCTION: Update state
        def update_state(State, Action_Sequence, Date, State_Variables):
            "Take the current state and recent history of actions and return the new state (clearing out the action sequence as necessary)"

            # If its the first day of the week, return the new state based on the actions from the previous week and reset the sequence of actions that we're keeping track of
            if pd.DatetimeIndex([Date])[0].dayofweek == 0:
                (Values, Labels) = make_states(Action_Sequence, False)
                Content = pd.DataFrame([Values.State[0]])
                Content.columns = Labels

                Container = pd.DataFrame([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).transpose()
                Container.columns = State_Variables

                New_State = tuple(pd.concat([Container, Content]).fillna(0).iloc[[1]].values[0])

                Action_Sequence = Action_Sequence.sort(columns=['Truck', 'Date'], ascending=False)
                Action_Sequence = Action_Sequence.drop_duplicates('Truck')

            # Else just update the day of week and quarter (I perfer to keep the state as a tuple generally so that I don't accidently change it)
            else:
                New_State = list(State)
                New_State[State_Variables.index('Day_Of_Week')] = pd.DatetimeIndex([Date])[0].dayofweek
                New_State[State_Variables.index('Quarter')] = pd.DatetimeIndex([Date])[0].quarter
                New_State = tuple(New_State)

            # Update global variables for debugging review
            global TIMES_CALLED
            TIMES_CALLED += 1
            global DATES
            DATES = DATES.append(pd.DataFrame([Date]))
            global STATES_REACHED
            STATES_REACHED = STATES_REACHED.append(pd.DataFrame([New_State]))

            return [New_State, Action_Sequence]

    # FUNCTION: Simulate a single path. ERROR: The first monday is not properly generating a variable... could be because I'm starting with a non-legit state
        def simulate_single_path(Probabilities, Starting_State, Starting_Date, Periods, Discount, State_Variables, Truck_ID, Action_Generator, Specific_Action):
            ""

            # Set the initial values
            Current_Date = dt.datetime.strptime(Starting_Date, '%Y-%m-%d')
            Current_State = Starting_State
            T = 0
            PDV_Profits = np.zeros(num_trucks)
            Action_Sequence = pd.DataFrame()

            while T < Periods:
                # Find the optimal actions and add to the action sequence            
                Actions = optimal_action(Probabilities, Current_State)

                # Replace specific truck's action with alternate strategy if requested
                if Action_Generator == 'Random':
                    Truck_Actions = generate_random_actions()
                    Truck_Actions = Truck_Actions[Truck_Actions.Truck == Truck_ID]
                    Actions = Actions[Actions.Truck != Truck_ID]
                    Actions = Actions.append(Truck_Actions)

                if Action_Generator == 'Specific':
                    Truck_Actions = generate_certain_actions(Specific_Action)
                    Truck_Actions = Truck_Actions[Truck_Actions.Truck == Truck_ID]
                    Actions = Actions[Actions.Truck != Truck_ID]
                    Actions = Actions.append(Truck_Actions)

                # Add the date to the current actions and append them to the action sequence
                Actions['Date'] = dt.datetime.strftime(Current_Date, '%Y-%m-%d')
                Action_Sequence = Action_Sequence.append(Actions)

                # Create the profit vector and add to the discounted sum of profits
                Period_Profits = create_profit_vector(State_Variables=State_Variables, State=Current_State, Actions=Actions)
                PDV_Profits += Discount**T*Period_Profits.Profit

                # Update state
                (Current_State, Action_Sequence) = update_state(State=Current_State, Action_Sequence=Action_Sequence, Date=dt.datetime.strftime(Current_Date, '%Y-%m-%d'), State_Variables=State_Variables)

                # Update counters
                T += 1
                Current_Date += dt.timedelta(days=1)

            return pd.DataFrame([Period_Profits.Truck, PDV_Profits]).transpose()

    # FUNCTION: Average over N simulations of the valuation function and (currently) return only for single truck
        def find_value_function(Probabilities, Starting_State, Starting_Date, Periods, Discount, State_Variables, Truck_ID, Action_Generator, Specific_Action, N):
            ""

            # Set initial values
            Value_Functions = np.zeros(num_trucks)

            # Run N times
            for x in xrange(N):
                Results = simulate_single_path(Probabilities=Probabilities, Starting_State=Starting_State, Starting_Date=Starting_Date, Periods=Periods, Discount=Discount, State_Variables=State_Variables, Truck_ID=Truck_ID, Action_Generator=Action_Generator, Specific_Action=Specific_Action)

                Value_Functions += 1./N * Results.Profit

            # Format results
            Step_One = pd.DataFrame([Results.Truck, Value_Functions]).transpose()

            return Step_One[Step_One.Truck == Truck_ID]

    (Locations, State_Variables) = make_states(Location_Data, True)
    test = find_value_function(Probabilities=find_probabilities(Locations), Starting_State=(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0), Starting_Date='2015-08-10', Periods=10, Discount=.99, State_Variables=State_Variables, Truck_ID=1, Action_Generator='Random', Specific_Action='', N=10)
    test.reset_index().Profit[0]



    # FUNCTION: Find g()


# FUNCTION: Estimate the parameters

# Run the program!

