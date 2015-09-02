#!/bin/bash

# Load python 
module load python/2.7-2015q2

# Execute the main run these should execute in series

# Create subfolder for the run
mkdir /home/eabrams/BBL/run0
cp locations.csv truck_types.csv $_ & wait
cd /home/eabrams/BBL/run0

# Run
python /home/eabrams/BBL/createdata.py
sbatch /home/eabrams/BBL/midway_simulate.sh
python /home/eabrams/BBL/estimate.py


# Will bootstrap later in a loop


# Will then manually combine the bootstrap
