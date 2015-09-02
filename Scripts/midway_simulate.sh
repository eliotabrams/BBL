#!/bin/bash

#SBATCH --job-name=arrayJob
#SBATCH --output=s%a.csv
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array=1-40
#SBATCH --time=24:00:00
#SBATCH --partition=sandyb
#SBATCH --ntasks=1


######################
# Begin work section #
######################

# Run simulation 40 times
python /home/eabrams/BBL/simulate.py
