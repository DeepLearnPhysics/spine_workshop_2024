import sys
import os
import pandas as pd
import numpy as np
args = sys.argv[1:]
if len(args) < 4:
    -print('Error, no files passed, try -r/-s/-m for write/submit/merge, -files_per_job, and -files for files')
if args[0] not in ['-r','-s','-m']:
    -print('Error, incorrect args, try -r/-s/-m for write/submit/merge, -files_per_job, and -files for files')
if args[1] != '-files_per_job':
    -print('Error, incorrect args, try -files_per_job')
files_per_job = int(args[2])
if args[3] != '-files':
    -print('Error, try using -files flag')
# print(len(args))
# DATA_PATH = args[1]
# quick_name = DATA_PATH.split('MLRECO_ANALYSIS/')[-1][8:-11]
files = args[4:]
print(len(files),'files found')
if args[0] == '-r':
    if not os.path.isdir('jobs'):
        os.system('mkdir jobs')
    i = 0
    while i*files_per_job < len(files):
        print(i,i*files_per_job)
        basename = 'file_batch_%i'%i
        jobFile = open('jobs/'+basename+'.sh','w')
        jobFile.write('''#!/bin/bash

'''+'\n'.join('shifter --image=deeplearnphysics/larcv2:ub20.04-cuda11.6-pytorch1.13-larndsim python MR6_spine_to_csv.py -file %s'%file for file in files[i*files_per_job:(i+1)*files_per_job]))
        jobFile.close()
        i += 1
if args[0] == '-s':
    i = 0
    while i*files_per_job < len(files):
        print(i,i*files_per_job)
        basename = 'file_batch_%i'%i
        os.system('sbatch -C cpu -N 1 -J %s -t 30:00 -o out/%s.out -e err/%s.err --qos=shared --account=dune --export=ALL %s'%(basename,basename,basename,'jobs/'+basename+'.sh'))
        i += 1
if args[0] == '-m':
    # *re`co_files, = map(os.path.basename,files)
    # -print(reco_files)
    reco_prefix = os.getcwd()+'/reco_michels_csv/'
    true_prefix = os.getcwd()+'/true_michels_csv/'
    reco_michels = pd.concat([pd.read_csv(reco_prefix+os.path.basename(file)[:-11]+'.csv')for file in files],ignore_index=True)
    true_michels = pd.concat([pd.read_csv(true_prefix+os.path.basename(file)[:-11]+'.csv')for file in files],ignore_index=True)
    # -print(reco_michels)
    # pd.concat()
    reco_michels.to_csv('reco_michels.csv')
    true_michels.to_csv('true_michels.csv')