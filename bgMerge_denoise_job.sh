#!/bin/csh 

hostname


# Usage:
# submitJob $launchd $workdir $gemcdir $coatdir $input $bgfile $gcard $yaml $nev 

set launchd  = $1
set workdir  = $2
set gemcdir  = $3
set coatdir  = $4
set input    = $5
set bgfile   = $6
set gcard    = $7
set yaml     = $8
set nev      = 1000
echo


set dir = $cwd
mkdir -p $dir
mkdir -p $dir/gemc
mkdir -p $dir/recon
mkdir -p $dir/reconBg


module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles  
module purge
module load clas12
#module switch coatjava/11.1.1
module load sqlite/dev
module load denoise/4.2.3

set cwd = `pwd`


		#cd /scratch/slurm/$SLURM_JOBID
		cd $cwd
		mkdir -p recon
		mkdir -p reconBg
		pwd
		ls -l

                if ( "`echo $input | cut -d'.' -f2`" == "hipo" )  then
			cp $input out.hipo
		else 
			echo
			echo Running gemc
			setenv GEMC_DATA_DIR $gemcdir
        	        cp $input input.lund
#			$gemcdir/source/gemc $gcard -USE_GUI=0 -INPUT_GEN_FILE="LUND, input.lund" -N=$nev
			gemc $gcard -USE_GUI=0 -INPUT_GEN_FILE="LUND, input.lund" -N=$nev
		endif

		echo
		echo "Updating bank schema"
		hipo-utils -update -d $coatdir/etc/bankdefs/hipo4 -o input.hipo out.hipo
		ls -l

		echo
		echo Starting processing input $input with bg file $bgfile
		cp $bgfile bg.hipo
		ls -l

                echo
                echo Running bg merging
                bg-merger bg.hipo -i input.hipo -o inputbg.hipo 
		ls -l

		echo
		echo "Running denoising"
		denoise2.exe  -i input.hipo -o denoised.hipo -l 0.01 -t 1
		ls -l

		echo
		echo Running reconstruction
		$coatdir/bin/recon-util -i denoised.hipo -o output.hipo -y $yaml 
		ls -l

		echo
		echo Filtering banks
                hipo-utils -filter -b "RUN*,REC*,MC*,DC*,Time**" -o recon/dst.hipo output.hipo
		ls -l recon/*
                rm output.hipo

                echo
                echo "Running denoising"
                denoise2.exe  -i inputbg.hipo -o denoisedbg.hipo -l 0.01 -t 1
                ls -l

                echo
                echo Running reconstruction
                $coatdir/bin/recon-util -i denoisedbg.hipo -o output.hipo -y $yaml 
                ls -l

                echo
                echo Filtering banks
                hipo-utils -filter -b "RUN*,REC*,MC*" -o reconBg/dst.hipo output.hipo
                ls -l reconBg/*
       	       	rm output.hipo input.hipo inputbg.hipo

		set simu = `basename $input | awk -F"." '{print $1}'`
		set simu = `basename $input | rev | cut -f 2- -d '.' | rev`		
		if ( "`echo $input | cut -d'.' -f2`" != "hipo" )  then
			mv out.hipo                   $dir/gemc/$simu.hipo
		endif
                mv recon/dst.hipo              $dir/recon/$simu.hipo
                mv reconBg/dst.hipo            $dir/reconBg/$simu.hipo

                ls -l

		rm -rf recon* input* bg.hipo out.hipo
		ls -l 
