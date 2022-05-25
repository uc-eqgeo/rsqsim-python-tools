#!/bin/bash
python catalogue_to_vtk.py --fault_model data/hik_creep.flt --catalogue_directory \
data --catalogue_prefix subduction --output_directory subduction_vtks \
--min_mw 7.5 --max_events 200