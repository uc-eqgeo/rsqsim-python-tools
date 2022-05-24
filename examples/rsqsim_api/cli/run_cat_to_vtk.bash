#!/bin/bash
python catalogue_to_vtk.py --fault_model ~/PycharmProjects/rsqsim-runs/subduction/hik_creep.flt --catalogue_directory \
~/PycharmProjects/rsqsim-runs/subduction --catalogue_prefix subduction --output_directory subduction_vtks \
--min_mw 7.5 --max_events 200