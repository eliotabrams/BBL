#!/bin/bash
#$ -j y

# Make folder
mkdir /home/eabrams0/run0
cp locations.csv truck_types.csv $_ & wait

# Create the bootstrap data
qsub -wd /home/eabrams0/run0 -q phd -N create_data0 /home/eabrams0/python_create_data.sh 

# Run the parallel simulations 
qsub -wd /home/eabrams0/run0 -q phd -N parallel0 -hold_jid create_data0 -t 1:3 /home/eabrams0/python_simulate_shell.sh

# Estimate the parameters 
qsub -wd /home/eabrams0/run0 -q phd -N estimate1 -hold_jid parallel0 /home/eabrams0/python_estimate.sh 


for i in {1..2}
do

    # Make folder
    mkdir /home/eabrams0/run$i
    cp locations.csv truck_types.csv $_ & wait

    # Create the bootstrap data
    qsub -wd /home/eabrams0/run$i -q phd -N create_data$i -hold_jid estimate$i /home/eabrams0/python_create_data.sh 

    # Run the parallel simulations 
    qsub -wd /home/eabrams0/run$i -q phd -N parallel$i -hold_jid create_data$i -t 1:3 /home/eabrams0/python_simulate_shell.sh

    # Estimate the parameters 
    qsub -wd /home/eabrams0/run$i -q phd -N estimate$((i+1)) -hold_jid parallel$i /home/eabrams0/python_estimate.sh 

done
