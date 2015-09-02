#!/bin/bash

# Eliot Abrams
# Shell to find point estimate and bootstrap SEs for BBL model

for i in {2..11}
do
    # Make folder for finding a bootstrapped point estimate
    mkdir /home/eabrams0/bootstrap$i
    cp locations.csv truck_types.csv $_ & wait

    # Create the bootstrap dataset
    qsub -wd /home/eabrams0/bootstrap$i -q phd -N create_data$i /home/eabrams0/python_create_bootstrap_data.sh 

done

for i in {2..11}
do
    # Run the simulations in parallel 
    qsub -wd /home/eabrams0/bootstrap$i -q phd -N parallel$i -hold_jid parallel$((i-1)) -t 1:50 /home/eabrams0/python_simulate_shell.sh

done

for i in {2..11}
do
    # Estimate the parameters 
    qsub -wd /home/eabrams0/bootstrap$i -q phd -hold_jid parallel$i /home/eabrams0/python_estimate.sh 

done



# Need to take the SEs of the bootstrap point estimates to finish
# Will do manually
