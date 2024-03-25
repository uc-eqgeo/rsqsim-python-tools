#!/bin/bash

python catalogue_to_vtk.py --fault_model /mnt/c/Users/jmc753/Work/RSQSim/Otago/merge_into_fault_model/otago_faults_2500_tapered_slip.flt \
--catalogue_directory /mnt/c/Users/jmc753/Work/RSQSim/Otago/otago-rsqsim-runs/otago_240214 --catalogue_prefix otago_1e6yr_nospinup \
--output_directory /mnt/c/Users/jmc753/Work/RSQSim/Otago/otago-rsqsim-runs/otago_240214/vtks --min_mw 6.5