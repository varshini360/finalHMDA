#!/bin/bash
#SBATCH --job-name=hmda_r2_test

#SBATCH --account=macs30123

#SBATCH --output=/home/varshi/finalHMDA/outputs/hmda_r2_test.out
#SBATCH --error=/home/varshi/finalHMDA/outputs/hmda_r2_test.err

#SBATCH --partition=caslake

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --time=01:00:00
#SBATCH --mem=32G

module load spark
module load python

cd /home/varshi/finalHMDA

spark-submit \
    --master local[8] \
    --driver-memory 24G \
    --executor-memory 24G \
    py_files/Hmda_r2.py
