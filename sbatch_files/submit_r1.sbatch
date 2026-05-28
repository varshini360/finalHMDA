#!/bin/bash
#SBATCH --job-name=hmda_r1_full

#SBATCH --account=macs30123

#SBATCH --output=/home/varshi/finalHMDA/outputs/hmda_r1full.out
#SBATCH --error=/home/varshi/finalHMDA/outputs/hmda_r1full.err

#SBATCH --partition=caslake

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --time=04:00:00
#SBATCH --mem=64G

module load spark
module load python

cd /home/varshi/finalHMDA

spark-submit \
    --master local[8] \
    --driver-memory 50G \
    --executor-memory 50G \
    py_files/Hmda_r1v2full.py
