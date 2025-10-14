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


/lustre24/expphy/volatile/clas12/bgualti/fall25/farmSubmit/submitJob /lustre24/expphy/volatile/clas12/bgualti/fall25/farmSubmit /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/gemc/5.11 /scigroup/cvmfs/hallb/clas12/sw/noarch/coatjava/11.1.0 /lustre24/expphy/volatile/clas12/bgualti/fall25/event_3.lund /work/osgpool/hallb/clas12/backgroundfiles/rgk_fall2018_FTOff/tor+1.00_sol-1.00/35nA_6535MeV/10k/00001.hipo /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/gemc/5.11/rgk_fall2018_FTOff.gcard /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/noarch/clas12-config/dev/coatjava/11.1.0/rgk_fall2018_FTOff.yaml 

