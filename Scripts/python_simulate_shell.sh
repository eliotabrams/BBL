#!/bin/bash
#$ -j y

/apps/bin/python27 /home/eabrams0/simulate.py > s$SGE_TASK_ID.csv

