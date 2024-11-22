#!/bin/bash

for FILE in /pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/0000000/MiniRun6_1E19_RHC.spine.*.SPINE.hdf5; do sbatch -C cpu -N 1 -J $(basename $FILE) -t 30:00 -o out/$(basename $FILE).out -e err/$(basename $FILE).err --qos=shared --account=dune --export=ALL ./run_script.sh $FILE; done
# for FILE in /pscratch/sd/d/dunepr/output/MiniRun6/run-spine/MiniRun6_1E19_RHC.spine/MLRECO_ANALYSIS/0000000/MiniRun6_1E19_RHC.spine.*.SPINE.hdf5; do source dummy_run_script.sh $FILE; done
# Replace all of above with a python script os.system(cmd)