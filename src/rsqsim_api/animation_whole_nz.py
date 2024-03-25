from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_background
from rsqsim_api.visualisation.animation import write_animation_frames, AnimateSequence
from rsqsim_api.io.rsqsim_constants import seconds_per_year


import os
import sys
import numpy as np
import shutil
from matplotlib import cm
import geopandas as gpd
import pandas as pd
import netCDF4 as nc
import argparse
from time import time


if __name__ == "__main__":
    # Checks to allow x1 to be a negative value (otherwise argparse thinks it's an argument flag)
    if '--bounds' in sys.argv:
        sysIx = sys.argv.index('--bounds') + 1
        if sys.argv[sysIx][0] == '-':
            sys.argv[sysIx] = ' ' + sys.argv[sysIx]

    parser = argparse.ArgumentParser(description="Arguments for animation")
    parser.add_argument("--procDir", help="Path to processing directory", type=str, default=os.path.join('/mnt', 'c', 'Users', 'jmc753', 'Work', 'RSQSim', 'Aotearoa', 'whole_nz_rsqsim'))
    parser.add_argument("--rsqsim_prefix", help="Prefix for RSQSim output files", type=str, default='whole_nz')
    parser.add_argument("--flt_file", help="Fault model .flt file in procDir", type=str, default='whole_nz_faults_2500_tapered_slip.flt')
    parser.add_argument("--noSerial", help="Data not in Serial type", default=True, action="store_false", dest='serial')
    parser.add_argument("--useFrames", help="Use Premade Frames", default=True, action="store_false", dest='remake_frames')
    parser.add_argument("--dispDir", help="Directory containing displacement maps (in procDir)", default='grds_5km', type=str)
    parser.add_argument("--min_t0", help="Start Time (yrs)", default=1e4, type=float)
    parser.add_argument("--max_t0", help="End Time (yrs)", default=2e4, type=float)
    parser.add_argument("--min_mw", help="Minimum Magnitude", default=7.0, type=float)
    parser.add_argument("--min_patches", help="Minimum number of patches", default=1, type=int)
    parser.add_argument("--hires_dem", help="Use high resolution DEM", default=False, action="store_true")
    parser.add_argument("--frameTime", help="Time between frames (yrs)", default=10, type=float)
    parser.add_argument("--max_crust_slip", help="Max plotted slip for crustal earthquakes", default=10, type=float)
    parser.add_argument("--max_sub_slip", help="Max plotted slip for subduction earthquakes", default=25, type=float)
    parser.add_argument("--max_disp_slip", help="Max plotted slip for displacement maps", default=1, type=float)
    parser.add_argument("--max_cum_slip", help="Max plotted slip for cumulative displacement maps", default=5, type=float)
    parser.add_argument("--frameRate", help="Frames per second", default=5, type=float)
    parser.add_argument("--displace", help="Include cumulative vertical displacements in the animation", default=False, action="store_true")
    parser.add_argument("--dispTimes", help="Times at which to plot cumulative displacements (2475 for 2 percent in 50 yrs) time1/time2", default='100/2475', type=str, dest='cum_times')
    parser.add_argument("--tideTime", help="Time span of tide gauge data", default=0, type=float, dest='tide_gauge_time')
    parser.add_argument("--tideLocation", help="Location of tide gauge x1/x2", default='wellington', type=str, dest='tide_gauge_location')
    parser.add_argument("--fadeTime", help="Number of seconds over which earthquakes fade", default=1, type=float)
    parser.add_argument("--fadePercent", help="Saturation that ruptures fade to (100 no fading, must be > 0)", default=4, type=float)
    parser.add_argument("--bounds", help="Grid Bounds x1/y1/x2/y2", default='1080000/4747500/2200000/6223500', type=str)
    parser.add_argument("--subd_plot", help="Plot subduction slip", default=False, action="store_true")
    parser.add_argument("--numThreads", help="Number of threads for plotting", default=1, type=int)
    parser.add_argument("--tideLim", help="Y-Limit for tide gauge plotting (m)", default=0.1, type=float)

    args = parser.parse_args()

    locations = {'dunedin': [1406510, 4917008],
                'wellington': [1749192, 5427448]}

    disp_map_dir = os.path.join(args.procDir, args.dispDir)

    if args.cum_times is not None:
        cum_times = [int(i) for i in args.cum_times.split('/')]
    else:
        cum_times = []

    if args.tide_gauge_location.lower() in locations.keys():
        tide_gauge_location = locations[args.tide_gauge_location.lower()]
    else:
        tide_gauge_location = [int(i) for i in args.tide_gauge_location.split('/')]

    bounds = [int(float(i)) for i in args.bounds.split('/')]

    if args.tide_gauge_time > 0 and not args.displace:
        args.displace = True

    # Fading variables
    fadeFrames = int(np.ceil(args.fadeTime * args.frameRate))  # number of frames over which an earthquake will fade
    time_to_threshold = fadeFrames * args.frameTime  # Time in years for fading to occur
    fading = np.power(100 / args.fadePercent, 1 / time_to_threshold)  # Fading rate required for plotting

    aniName = '{}_{:.1e}-{:.1e}_Mw{:.1f}_{}yr'.format(args.rsqsim_prefix, args.min_t0, args.max_t0, args.min_mw, args.frameTime)
    aniName = aniName.replace('+', '')

    if args.displace:
        frameTime = [args.frameTime] + cum_times
    else:
        frameTime = [args.frameTime]

    tide = {'time': int(frameTime[0] * np.ceil(args.tide_gauge_time / frameTime[0])), 'x': tide_gauge_location[0], 'y': tide_gauge_location[1], 'ylim': args.tideLim}

    print('Earthquakes over Mw {} to fade over {} seconds (i.e. {} frames covering {} years at {} fps)'.format(args.min_mw, args.fadeTime, fadeFrames, time_to_threshold, args.frameRate))
    print('Movie Name: {}.mp4\n'.format(aniName))

    if args.subd_plot:
        max_sub_slip = args.max_sub_slip
    else:
        max_sub_slip = 0

    eq_output_file = os.path.join(args.procDir, 'eqs.' + args.rsqsim_prefix + '.out')
    flt_file = os.path.join(args.procDir, args.flt_file)

    animationDir = os.path.join(args.procDir, aniName)
    frameDir=os.path.join(animationDir, "frames")
    if  not os.path.exists(animationDir):
        os.mkdir(animationDir)

    if  os.path.exists(frameDir) and args.remake_frames:
        shutil.rmtree(frameDir)  # Remove old frames
        os.mkdir(frameDir)
    elif not os.path.exists(frameDir):
        os.mkdir(frameDir)
        if not args.remake_frames:
            print('No frameDir existed. Setting remake_frames to True.')
            args.remake_frames = True
        

    trimname = "trimmed_" + args.rsqsim_prefix

    catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(eq_output_file, args.procDir, args.rsqsim_prefix, serial=args.serial)

    trimmed_catalogue = catalogue.filter_whole_catalogue(min_t0=args.min_t0 * seconds_per_year, max_t0=args.max_t0 * seconds_per_year,
                                                        min_mw=args.min_mw)
    trimmed_catalogue.write_csv_and_arrays(trimname, directory=animationDir)

    # Read in the trimmed faults

    if __name__ == "__main__":
        trimmed_faults = RsqSimMultiFault.read_fault_file_keith(fault_file=flt_file)
        trimmed_catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(animationDir,trimname))
        filtered_events = trimmed_catalogue.events_by_number(trimmed_catalogue.catalogue_df.index, trimmed_faults, min_patches=args.min_patches)

        if tide['time'] > 0:
            print('Calculating synthetic tide gauge')
            frame_times = np.arange(args.min_t0, args.max_t0 + frameTime[0], frameTime[0])
            event_df = pd.read_csv(os.path.join(animationDir,trimname + '_catalogue.csv'))
            TG = np.zeros((len(frame_times), 3))  # Frame ID, Year, Tide Level
            TG[:, 0] = np.arange(len(frame_times))
            TG[:, 1] = frame_times
            frame_times = frame_times * seconds_per_year
            for frame in range(1, len(frame_times)):
                events = event_df[event_df['t0'].between(frame_times[frame - 1], frame_times[frame])]
                disp = TG[frame - 1, 2]
                if events.shape[0] > 0:
                    first_event = True  # Flag for first event in time period - needed for searching for available displacement maps
                    for ix, event in enumerate(events[events.columns[0]].values):
                        if os.path.exists(os.path.join(disp_map_dir, f"ev{event:.0f}.grd")):
                            disp_grd = nc.Dataset(os.path.join(disp_map_dir, f"ev{event:.0f}.grd"))
                            if first_event:
                                dispX = disp_grd['x'][:].data
                                dispY = disp_grd['y'][:].data
                                dx = np.diff(dispX)[0]
                                dy = np.diff(dispY)[0]
                                gridX = np.round((tide['x'] - dispX[0]) / dx)
                                gridY = np.round((tide['y'] - dispY[0]) / dy)
                                first_event = False
                            disp = np.nansum([disp, disp_grd['z'][int(gridY), int(gridX)].data])
                TG[frame, 2] = disp
            tide['entries'] = int(tide['time'] / frameTime[0])  # Number of entries in each tide guage time series
            tide['file'] = os.path.join(animationDir, 'tide_gauge.npy')
            tide['data'] = TG
            np.save(tide['file'], TG)   

        if args.remake_frames:
            begin = time()
            print('Plotting Background')
            background = plot_background(plot_lakes=False, bounds=bounds,
                                    plot_highways=False, plot_rivers=False, hillshading_intensity=0.3,
                                    pickle_name=os.path.join(animationDir,'temp.pkl'), hillshade_cmap=cm.Greys, hillshade_fine=args.hires_dem,
                                    plot_edge_label=False, figsize=(10, 10), plot_sub_cbar=args.subd_plot, plot_crust_cbar=True,
                                    slider_axis=True, crust_slip_max=args.max_crust_slip, sub_slip_max=max_sub_slip,
                                    displace=args.displace, disp_slip_max=args.max_disp_slip, cum_slip_max=args.max_cum_slip, step_size=frameTime, tide=tide)

            print('Plotting animation frames')
            write_animation_frames(args.min_t0, args.max_t0, frameTime, trimmed_catalogue, trimmed_faults,
                            pickled_background=os.path.join(animationDir,'temp.pkl'), bounds=bounds,
                            extra_sub_list=["hikurangi", "hikkerm", "puysegur"], time_to_threshold=time_to_threshold,
                            global_max_sub_slip=max_sub_slip, global_max_slip=args.max_crust_slip, min_mw=args.min_mw, decimals=0,
                            fading_increment=fading, frame_dir=frameDir, num_threads_plot=args.numThreads, min_slip_value=0.2,
                            displace=args.displace, disp_slip_max=args.max_disp_slip, cum_slip_max=args.max_cum_slip, 
                            disp_map_dir=disp_map_dir, tide=tide)
            print('Frames plotted in {:.2f} seconds'.format(time() - begin))
        else:
            print('Reusing previous frames')

        print('\nStitching frames into animation')

        aniName = '{}_{:.0e}-{:.0e}_Mw{:.1f}_{}yr'.format(args.rsqsim_prefix, args.min_t0, args.max_t0, args.min_mw, frameTime[0])
        aniName = aniName.replace('+', '')

        ffmpeg = "ffmpeg -framerate {2} -i '{0:s}/frame%04d.png' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -y '{1:s}/{3:s}.mp4'".format(frameDir, animationDir, args.frameRate, aniName)
        print(ffmpeg, '\n')
        os.system(ffmpeg)
        os.system('cat {} > movieMakerCmd.txt'.format(ffmpeg))

        # Tidy up
        tidy = True
        filesuffixes = ['events', 'patches', 'slip', 'slip_time']
        if tidy:
            os.remove(os.path.join(animationDir,'temp.pkl'))
            for junk in filesuffixes:
                os.remove(os.path.join(animationDir,'{}_{}.npy'.format(trimname, junk)))