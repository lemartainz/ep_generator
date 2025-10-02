#!/bin/bash
#SBATCH --job-name=clas12_recon
#SBATCH --output=clas12_recon_%j.out
#SBATCH --error=clas12_recon_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=production
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load required modules
module purge
module switch -f gemc/5.11
module load coatjava/11.1.0

# Input and output files
LUND_FILE="events.lund"
GCARD="/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/gemc/5.11/rgk_fall2018_FTOff.gcard"
YAML="/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/coatjava/11.1.0/rgk_fall2018_FTOff.yaml"

# Run GEMC simulation
gemc $GCARD -USE_GUI=0 -INPUT_GEN_FILE="LUND, ${LUND_FILE}" -N=5

# Rename default GEMC output
mv out.hipo out_slurm.hipo

# Run Reconstruction
recon-util -i out_slurm.hipo -o recon_slurm.hipo -y $YAML

# Filter banks to keep only RUN::config, MC::*, REC::*
hipo-utils -filter -b "RUN::config,MC::*,REC::*" -o recon_selectedBanks.hipo recon_slurm.hipo

echo "GEMC, reconstruction, and bank selection complete!"
