#!/bin/bash
#SBATCH --job-name=hmda_r1

#SBATCH --account=macs30123

#SBATCH --output=/home/varshi/hmda/outputs/hmda_r1.out
#SBATCH --error=/home/varshi/hmda/outputs/hmda_r1.err

#SBATCH --partition=caslake

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

#SBATCH --time=01:00:00
#SBATCH --mem=16G

module load spark
module load python

cd /home/varshi/hmda

spark-submit \
    --master local[4] \
    --driver-memory 12G \
    --executor-memory 12G \
    py_files/Hmda_r1v2.py
