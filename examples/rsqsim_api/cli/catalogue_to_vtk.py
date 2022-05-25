import os.path
from os import mkdir
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Required arguments to extract slip distributions")
    parser.add_argument("--fault_model", required=True, help="Path to fault model", action="store")
    parser.add_argument("--catalogue_directory", required=True, help="Directory containing files", type=str)
    parser.add_argument("--catalogue_prefix", help="prefix of catalogue and list files", action="store",
                        required=True, type=str)
    parser.add_argument("--output_directory", "-o", required=True, help="Directory to store output vtks", action="store")
    parser.add_argument("--min_mw", help="Filter catalogue to exclude events smaller than magnitude mw",
                        action="store", type=float, default=None)
    parser.add_argument("--max_events", help="Write only first N events", action="store", type=int, default=None)

    args = parser.parse_args()

    fault_model = RsqSimMultiFault.read_fault_file_keith(args.fault_model)
    whole_catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(args.catalogue_directory, args.catalogue_prefix))

    if args.min_mw is not None:
        filtered_catalogue = whole_catalogue.filter_whole_catalogue(min_mw=args.min_mw)
    else:
        filtered_catalogue = whole_catalogue

    if not os.path.exists(args.output_directory):
        mkdir(args.output_directory)

    if args.max_events is not None:
        short_catalogue = filtered_catalogue.filter_by_events(filtered_catalogue.catalogue_df.index[:args.max_events])
    else:
        short_catalogue = filtered_catalogue

    short_catalogue.all_slip_distributions_to_vtk(fault_model, args.output_directory)
    short_catalogue.catalogue_df.to_csv(os.path.join(args.output_directory, "events.csv"), index=True)

    





