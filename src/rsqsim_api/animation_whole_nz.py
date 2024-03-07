from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_background
from rsqsim_api.visualisation.animation import write_animation_frames, AnimateSequence
from rsqsim_api.io.rsqsim_constants import seconds_per_year


import os
import numpy as np
import shutil
from matplotlib import cm
import geopandas as gpd


# Run Name
rsqsim_list_file_prefix = 'otago_1e6yr'

# Define Variables
procDir = os.path.join('/mnt', 'c', 'Users', 'jmc753', 'Work', 'RSQSim', 'Otago', 'otago-rsqsim-runs', 'otago_240214')  # Main processing directory

flt_file = 'otago_faults_2500_tapered_slip.flt'

trimname = []  # Name for trimmed EQ files
serial = False

# Trim Catalogue Variables
min_t0 = 1e5 # Start Time (yrs)
max_t0 = 2e5 # End Time (yrs)
min_mw = 7.0 # Minimum Magnitude
min_patches = 1

# Plotting variables
step_size = 50  # Years between frames
max_crust_slip = 10  # Max colourbar limit
max_sub_slip = 25  # Max colourbar limit
framerate = 100  # Frames per second
remake_frames = True  # Remake frames if they already exist
hires_dem = True  # Use high resolution DEM


# Fading variables
fadingTime = 1  # number of seconds over which an earthquake will fade
fadingPercent = 4  # Saturation that ruptures fade to (100 no fading, must be > 0)
fadeFrames = int(np.ceil(fadingTime * framerate))  # number of frames over which an earthquake will fade
time_to_threshold = fadeFrames * step_size  # Time in years for fading to occur
fading = np.power(100/fadingPercent, 1/time_to_threshold)  # Fading rate required for plotting

aniName = '{}_{:.0e}-{:.0e}_Mw{:.1f}_{}yr'.format(rsqsim_list_file_prefix,min_t0, max_t0, min_mw, step_size)
aniName = aniName.replace('+', '')

print('Earthquakes over Mw {} to fade over {} seconds (i.e. {} frames covering {} years at {} fps)'.format(min_mw, fadingTime, fadeFrames, time_to_threshold, framerate))
print('Movie Name: {}.mp4\n'.format(aniName))

# Image bounds in NZTM
min_x1 = 1150000
min_y1 = 4800000
max_x2 = 1450000
max_y2 = 5100000
bounds = [800000, 4000000, 3200000, 7000000]  # National Coverage
bounds = [min_x1, min_y1, max_x2, max_y2]

subd_plot = False
if subd_plot: max_sub_slip = 0

eq_output_file = os.path.join(procDir, 'eqs.' + rsqsim_list_file_prefix + '.out')
flt_file = os.path.join(procDir, flt_file)

animationDir = os.path.join(procDir, aniName)
frameDir=os.path.join(animationDir, "frames")
if  not os.path.exists(animationDir):
    os.mkdir(animationDir)

if  os.path.exists(frameDir) and remake_frames:
    shutil.rmtree(frameDir)  # Remove old frames
    os.mkdir(frameDir)
elif not os.path.exists(frameDir):
    os.mkdir(frameDir)
    if not remake_frames:
        print('No frameDir existed. Setting remake_frames to True.')
        remake_frames = True
    

if not trimname: trimname = "trimmed_" + rsqsim_list_file_prefix

catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(eq_output_file, procDir, rsqsim_list_file_prefix, serial=serial)

trimmed_catalogue = catalogue.filter_whole_catalogue(min_t0=min_t0 * seconds_per_year, max_t0=max_t0 * seconds_per_year,
                                                     min_mw=min_mw)
trimmed_catalogue.write_csv_and_arrays(trimname, directory=animationDir)

# Read in the trimmed faults

if __name__ == "__main__":
    trimmed_faults = RsqSimMultiFault.read_fault_file_keith(fault_file=flt_file)
    trimmed_catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(animationDir,trimname))
    filtered_events = trimmed_catalogue.events_by_number(trimmed_catalogue.catalogue_df.index, trimmed_faults, min_patches=min_patches)

    print('Plotting Background')
    background = plot_background(plot_lakes=False, bounds=bounds,
                                plot_highways=False, plot_rivers=False, hillshading_intensity=0.3,
                                 pickle_name=os.path.join(animationDir,'temp.pkl'), hillshade_cmap=cm.Greys, hillshade_fine=hires_dem,
                                 plot_edge_label=False, figsize=(10, 10), plot_sub_cbar=subd_plot, plot_crust_cbar=True,
                                 slider_axis=True, crust_slip_max=max_crust_slip, sub_slip_max=max_sub_slip)

    if remake_frames:
        print('Plotting animation frames')
        write_animation_frames(min_t0, max_t0, step_size, trimmed_catalogue, trimmed_faults,
                           pickled_background=os.path.join(animationDir,'temp.pkl'), bounds=bounds,
                           extra_sub_list=["hikurangi", "hikkerm", "puysegur"], time_to_threshold=time_to_threshold,
                           global_max_sub_slip=max_sub_slip, global_max_slip=max_crust_slip, min_mw=min_mw, decimals=0,
                           fading_increment=fading, frame_dir=frameDir, num_threads_plot=None, min_slip_value=0.2)
    else:
        print('Reusing previous frames')

    print('\nStitching frames into animation')

    aniName = '{}_{:.0e}-{:.0e}_Mw{:.1f}_{}yr'.format(rsqsim_list_file_prefix,min_t0, max_t0, min_mw, step_size)
    aniName = aniName.replace('+', '')

    ffmpeg = "ffmpeg -framerate {2} -i '{0:s}/frame%04d.png' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -y '{1:s}/{3:s}.mp4'".format(frameDir, animationDir, framerate, aniName)
    print(ffmpeg, '\n')
    os.system(ffmpeg)

    # Tidy up
    tidy = True
    filesuffixes = ['events', 'patches', 'slip', 'slip_time']
    if tidy:
        os.remove(os.path.join(animationDir,'temp.pkl'))
        for junk in filesuffixes:
           os.remove(os.path.join(animationDir,'{}_{}.npy'.format(trimname, junk)))