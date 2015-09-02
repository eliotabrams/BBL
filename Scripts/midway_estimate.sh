#!/bin/bash

#SBATCH --job-name=arrayJob
#SBATCH --output=o_estimate.csv
#SBATCH --error=e_estimate.err
#SBATCH --time=24:00:00
#SBATCH --partition=sandyb
#SBATCH --ntasks=1


######################
# Begin work section #
######################

# Load python 
module load python/2.7-2015q2

# Run estimation
python /home/eabrams/BBL/estimate.py
