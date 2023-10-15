#!/bin/sh
#SBATCH -p ProdQ
#SBATCH -N 40
#SBATCH -t 72:00:00
#SBATCH --ntasks=40
# Charge job to my account 
#SBATCH -A nuim01
# Write stdout+stderr to file
#SBATCH -o output.txt
#SBATCH --mail-user=fred.valdezameneyro.2019@mumail.ie
#SBATCH --mail-type=BEGIN,END

module load taskfarm
taskfarm carcassonne_1p.txt
