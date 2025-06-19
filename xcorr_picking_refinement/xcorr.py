import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

import os, glob, warnings
from collections import defaultdict as dfdict
import pandas as pd
from obspy import UTCDateTime, read, Stream, Trace
from obspy.signal.cross_correlation import xcorr_pick_correction
import random
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")

# define a nested defaultdict 
def nested_dict():
    return dfdict(nested_dict)
    
# get the nested value keys
def get_nested_dict(d):
    keys = []
    for value in d.values():
        if isinstance(value, dict):
            keys.extend(value.keys())
    return keys

def read_wave(sta_dict, wave_path, date):
    # get the time reference for trimming
    time_ref = UTCDateTime(sta_dict[list(sta_dict.keys())[0]]['P'])
    stream = Stream()
    for sta in sta_dict.keys():
        for w in glob.glob(os.path.join(wave_path.joinpath(f"{sta}"), f"*{date}*"), recursive = True):
            try:
                stread = read(w)
                stream += stread
            except Exception as e:
                continue
    stream.trim(time_ref - 5, time_ref + 5)
    return stream
        
def start_correlation (pick_dir, xcorr_data, wave_form):
    h5 = h5py.File(xcorr_data, 'r')
    xcorr_list = [[name.split('.')[0], name] for name in h5['data'].keys()] 
    xcorr_name_df = pd.DataFrame(xcorr_list, columns=["Stations", "File_name"])
    FilesName = glob.glob(os.path.join(pick_dir,'*.pick'), recursive = True)
    with tqdm (total = len(FilesName)) as pbar:
        for file in FilesName:
            date = Path(file).stem[:8]
            lines = [l.split() for l in open(file, 'r')]
            write_rev_pick = open(pick_dir.joinpath(f"{Path(file).stem}_rev.pick"), 'w')
        
            # initialize data frame holder
            # initialize dict
            holder = nested_dict()
            for line in lines:
                pick_time = (f"{line[6][:4]}-{int(line[6][4:6]):02d}-{int(line[6][6:8]):02d}T"
                        f"{int(line[7][:2]):02d}:{int(line[7][2:]):02d}:{float(line[8]):08.5f}")
                #holder[f"{line[0]}_{line[4]}"] = pick_time
                holder[f"{line[0]}"][f"{line[4]}"] = pick_time
        
            # read and trim the waveform
            stream_read = read_wave(holder, wave_form, date)
            
            for sta in holder.keys():
                # select all stations from mainstream
                stream2 = stream_read.select(station = sta)
                
                phases = holder[sta].keys()
                for phase in phases:
                    pick_time = UTCDateTime(holder[sta][phase])
                    selected_staxcorr = xcorr_name_df[xcorr_name_df['Stations'] == sta]
                    random_sel = random.sample(list(selected_staxcorr['File_name']), k = 15)
                    
                    # start correlation
                    best_coeff = -np.inf
                    for xcorr_master in random_sel:
                        # load the master file
                        master_trace = h5['data'][f"{xcorr_master}"]
                        if phase == 'P':
                            # get the master trace data
                            master_time = UTCDateTime(master_trace.attrs["p_time"])
                            tr_Z_mtd = json.loads(master_trace.attrs['chan_Z_metadata'])
                            trace_z_master = Trace(np.array([i[2] for i in master_trace]), header = tr_Z_mtd)
                            
                            # get the pair trace data
                            trace_z_pair = stream2.select(component = 'Z')[0]
            
                            try:
                                dt, coeff = xcorr_pick_correction(master_time, trace_z_master, pick_time, trace_z_pair, 0.2, 0.2, 0.4, plot=False)
                            except Exception:
                                continue
                            
                            if coeff > best_coeff:
                                comp = 'BHZ'
                                best_coeff = coeff
                                correction_time = dt
                                pick_attr = master_trace.attrs["p_attrs"]
                            
                                
                        elif phase == 'S':
                            # get the master trace data
                            master_time = UTCDateTime(master_trace.attrs["s_time"])
                            tr_N_mtd = json.loads(master_trace.attrs['chan_N_metadata'])
                            trace_n_master = Trace(np.array([i[1] for i in master_trace]), header = tr_N_mtd)
            
                            # get the pair trace data
                            trace_n_pair = stream2.select(component = 'N')[0]
            
                            # do cross correlation
                            try:
                                dt, coeff = xcorr_pick_correction(master_time, trace_n_master, pick_time, trace_n_pair, 0.2, 0.2, 0.4, plot=False)
                            except Exception:
                                continue
                            if coeff > best_coeff:
                                comp = 'BHN'
                                best_coeff = coeff
                                correction_time = dt
                                pick_attr = master_trace.attrs["s_attrs"]
                    pick_corrected = pick_time + correction_time
                    write_rev_pick.write(
                        (f"{sta} ? {comp} {pick_attr} {pick_corrected.year:}{pick_corrected.month:02d}{pick_corrected.day:02d} "
                         f"{pick_corrected.hour:02d}{pick_corrected.minute:02d} {float((pick_corrected.second + (pick_corrected.microsecond/1e6))):08.5f} "
                         f"GAU 0.0 0.0 0.0 0.0\n")  
                    )
                    write_rev_pick.flush()
            write_rev_pick.close()
            pbar.update()
    return None
 
def main():
    # dir list
    pick_file = Path(r"C:\Users\User\eqt-project\pick_2023_09")
    h5_file = Path(r"C:\Users\User\eqt-project\cross-correlation\xcorr.hdf5")
    wave = Path(r"C:\Users\User\eqt-project\ai_seml_3")
    
    start_correlation(pick_file, h5_file, wave)
    return None

if __name__ == "__main__":
    main()
