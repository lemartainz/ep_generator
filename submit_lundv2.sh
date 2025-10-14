#!/bin/bash
#SBATCH --job-name=TEST_eFD_65_BGmerge_1000evts
#SBATCH --account=clas12
#SBATCH --mail-user=bgualti@jlab.org
#SBATCH --partition=priority
#SBATCH --output=%x-%j-%N.out
#SBATCH --error=%x-%j-%N.err
#SBATCH --gres=disk:5000
#SBATCH --mem-per-cpu=1800
#SBATCH --time=4:00:00


# Load required modules
module purge
module switch -f gemc/5.11
module load coatjava/11.1.0

# Input and output files
LUND_FILE="event_3.lund"
GCARD="/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/gemc/5.11/rgk_fall2018_FTOff.gcard"
YAML="/cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/coatjava/11.1.0/rgk_fall2018_FTOff.yaml"
BG_FILE="/work/osgpool/hallb/clas12/backgroundfiles/rgk_fall2018_FTOff/tor+1.00_sol-1.00/35nA_6535MeV/10k/00001.hipo"

# Run GEMC simulation
gemc $GCARD -USE_GUI=0 -INPUT_GEN_FILE="LUND, ${LUND_FILE}" -N=1000

# Rename default GEMC output
mv out.hipo out_slurm.hipo

# --- Merge background file with GEMC output ---
echo "Merging background file..."
bg-merger -b ${BG_FILE} -i out_slurm.hipo -o out_merged.hipo -d "DC,FTOF,ECAL,HTCC,LTCC,BST,BMT,CND,CTOF,FTCAL,FTHODO"

# Run Reconstruction on the merged file
recon-util -i out_merged.hipo -o recon_slurm.hipo -y $YAML

# Filter banks to keep only RUN::config, MC::*, REC::*
hipo-utils -filter -b "RUN::config,MC::*,REC::*" -o recon_selectedBanks.hipo recon_slurm.hipo

echo "GEMC, background merging, reconstruction, and bank selection complete!"

START=$(date +%s)

# --- your GEMC and reconstruction commands here ---

END=$(date +%s)
ELAPSED=$((END - START))
echo "Total runtime: $((ELAPSED / 3600)) hours and $(((ELAPSED % 3600) / 60)) minutes."
