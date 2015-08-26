#!/bin/bash

# Eliot Abrams
# Shell to find point estimate and bootstrap SEs for BBL model


# Make folder for finding point estimate
mkdir /home/eabrams0/run0
cp locations.csv truck_types.csv $_ & wait

# Create the original dataset
qsub -wd /home/eabrams0/run0 -q phd -N create_data0 /home/eabrams0/python_create_data.sh 

# Run the simulations in parallel 
qsub -wd /home/eabrams0/run0 -q phd -N parallel0 -hold_jid create_data0 -t 1:100 /home/eabrams0/python_simulate_shell.sh

# Estimate the parameters 
qsub -wd /home/eabrams0/run0 -q phd -N estimate1 -hold_jid parallel0 /home/eabrams0/python_estimate.sh 


for i in {1..10}
do

    # Make folder for finding a bootstrapped po
    mkdir /home/eabrams0/bootstrap$i
    cp locations.csv truck_types.csv $_ & wait

    # Create the bootstrap data
    qsub -wd /home/eabrams0/bootstrap$i -q phd -N create_data$i -hold_jid estimate$i /home/eabrams0/python_create_bootstrap_data.sh 

    # Run the simulations in parallel 
    qsub -wd /home/eabrams0/bootstrap$i -q phd -N parallel$i -hold_jid create_data$i -t 1:100 /home/eabrams0/python_simulate_shell.sh

    # Estimate the parameters 
    qsub -wd /home/eabrams0/bootstrap$i -q phd -N estimate$((i+1)) -hold_jid parallel$i /home/eabrams0/python_estimate.sh 

done

# Need to take the SEs of the bootstrap point estimates to finish
# Will do manually
