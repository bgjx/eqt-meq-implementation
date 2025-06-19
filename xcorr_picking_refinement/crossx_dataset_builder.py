# created dataset for noise data
import os
import h5py
import csv
import numpy as np
from scipy.signal import convolve
import glob
import pandas as pd
import time
import re
from tqdm import tqdm
from time import sleep
from datetime import datetime
import json

import obspy
from obspy import UTCDateTime, read, Stream
from obspy.geodetics import gps2dist_azimuth
import matplotlib.pyplot as plt
from pathlib import Path


def build (hypo_cat, picking_cat, station, wave_form, output, hdf_out_name):

    # read the data
    hypo_df = pd.read_excel(hypo_cat, index_col = None)
    pick_df = pd.read_excel(picking_cat, index_col = None)
    #sta_df  = pd.read_excel(station, index_col = None)

    # iterate through the hypo dataframe 
    id_start = 3094
    id_end = 3295
    # id_end = 3295

    # id list for selected events only
    id_list = [3575,3604,3611,3612,3614,3615,3616,3618,3620,3636,3645,3646,3653]

    # initialize dataset
    HDF0 = h5py.File(hdf_out_name, 'a')
    HDF0.create_group("data")

    # generate impulse kernel for synthetic noise

    # Define a kernel (e.g., a simple sinusoidal impulse response)
    kernel_length = 30  # Length of the kernel
    kernel_t = np.linspace(0, kernel_length / 100, kernel_length, endpoint=False)
    frequency = 7.5  # Frequency of the kernel (Hz)
    amplitude = 0.8  # Amplitude of the kernel

    # Generate the kernel (impulse response)
    kernel = amplitude * np.sin(2 * np.pi * frequency * kernel_t) * np.exp(-kernel_t / 0.1)  # Sine wave with exponential decay

    with tqdm (total = (id_end - id_start) +1) as pbar:
        for _id in range(id_start, id_end +1):
    #     for _id in id_list:
            pbar.update()
            
            # select the data
            hypo_dat = hypo_df[hypo_df.ID == _id]

            # filter the unwanted data
            try:
                if hypo_dat.Elev.iloc[0] > 1000:
                    continue
                if hypo_dat.Remarks.iloc[0] == 'Initial' and hypo_dat.Mw_mag.iloc[0] < 0:
                    continue
                if hypo_dat.Remarks.iloc[0] == 'Reloc' and hypo_dat.Mw_mag.iloc[0] < 0:
                    continue
            except Exception:
                continue
            
            # select the picking data
            pick_dat = pick_df[pick_df['Event ID'] == _id]
            
            # get the waveform
            stream = Stream()
            for w in glob.glob(os.path.join(wave_form.joinpath(f"{_id}"), '*.mseed'), recursive = True):
                try:
                    stread = read(w)
                    stream += stread
                except Exception as e:
                    logger.error(f"Error reading waveform {w}: {e}") 
            sst = stream.copy()

            for sta in pick_dat.get('Station'):
                
                # get arrival time info
                pick_info = pick_dat[pick_dat.Station == sta].iloc[0]
                p_arrival = UTCDateTime(
                    f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
                    f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_P):02d}:{float(pick_info.P_Arr_Sec):012.9f}")
                s_arrival = UTCDateTime(
                    f"{pick_info.Year}-{int(pick_info.Month):02d}-{int(pick_info.Day):02d}T"
                    f"{int(pick_info.Hour):02d}:{int(pick_info.Minutes_S):02d}:{float(pick_info.S_Arr_Sec):012.9f}")

                # P onset attribut
                amp_onset = pick_info.P_Polarity
                if amp_onset == '+':
                    amp_onset = 'c'
                elif amp_onset == '-':
                    amp_onset = 'd'
                else:
                    amp_onset = '?'
                    
                clarity = pick_info.P_Onset
                if clarity == 'I':
                    clarity = 'i'
                elif clarity == 'E':
                    clarity = 'e'
                else:
                    clarity = '?'

                P_attr = f"{clarity} P {amp_onset}"
                S_attr = 'e S ?'
                
                # read the waveform
                try:
                    traces = sst.select(station = sta)
                    stat_trace = traces[0].stats
                except IndexError:
                    continue
                    
                starttime = stat_trace.starttime
                sampling_rate = stat_trace.sampling_rate

                # calculate p and s arrival as the wave sample
                p_arr_sample = int((p_arrival - starttime)*sampling_rate) 
                s_arr_sample = int((s_arrival - starttime)*sampling_rate)

                # create trace name
                trace_name = (f"{sta}.{'ML'}_"
                              f"{starttime.year}{starttime.month:>02}{starttime.day:>02}T{starttime.hour:>02}{starttime.minute:>02}{starttime.second:>02}_EV")
                
                try:
                    # creating dataset to be stored in hd5f file format
                    Tr_E = traces.select(component = 'E')[0]
                    Tr_E.data = Tr_E.data[:3000]
                    Tr_E_stats =         {
                        'network': Tr_E.stats.network,
                        'station': Tr_E.stats.station,
                        'location':Tr_E.stats.location ,
                        'channel': Tr_E.stats.channel,
                        'starttime': f"{Tr_E.stats.starttime}",
                        'endtime': f"{Tr_E.stats.endtime}",
                        'sampling_rate': Tr_E.stats.sampling_rate,
                        'delta': Tr_E.stats.delta,
                        'npts': Tr_E.stats.npts,
                        'calib': Tr_E.stats.calib,
                        '_format': Tr_E.stats._format,
                        'mseed': f"{Tr_E.stats.mseed}"
                    }
                    Tr_E_mtd = json.dumps(Tr_E_stats)
                    
                    Tr_N = traces.select(component = 'N')[0]
                    Tr_N.data = Tr_N.data[:3000]
                    Tr_N_stats =         {
                        'network': Tr_N.stats.network,
                        'station': Tr_N.stats.station,
                        'location':Tr_N.stats.location ,
                        'channel': Tr_N.stats.channel,
                        'starttime': f"{Tr_N.stats.starttime}",
                        'endtime': f"{Tr_N.stats.endtime}",
                        'sampling_rate': Tr_N.stats.sampling_rate,
                        'delta': Tr_N.stats.delta,
                        'npts': Tr_N.stats.npts,
                        'calib': Tr_N.stats.calib,
                        '_format': Tr_N.stats._format,
                        'mseed': f"{Tr_N.stats.mseed}"
                        }
                    Tr_N_mtd = json.dumps(Tr_N_stats)
                    
                    Tr_Z = traces.select(component = 'Z')[0]
                    Tr_Z.data = Tr_Z.data[:3000]
                    Tr_Z_stats =         {
                        'network': Tr_Z.stats.network,
                        'station': Tr_Z.stats.station,
                        'location':Tr_Z.stats.location ,
                        'channel': Tr_Z.stats.channel,
                        'starttime': f"{Tr_Z.stats.starttime}",
                        'endtime': f"{Tr_Z.stats.endtime}",
                        'sampling_rate': Tr_Z.stats.sampling_rate,
                        'delta': Tr_Z.stats.delta,
                        'npts': Tr_Z.stats.npts,
                        'calib': Tr_Z.stats.calib,
                        '_format': Tr_Z.stats._format,
                        'mseed': f"{Tr_Z.stats.mseed}"
                        }
                    Tr_Z_mtd = json.dumps(Tr_Z_stats)
                except Exception as e:
                    print(e)
                    continue
                
                #calculate the coda end sample
                coda_end_sample = s_arr_sample + int(((s_arr_sample - p_arr_sample))) + 500 # add 5 second
                coda_end_sample_arr = np.array([[coda_end_sample]])
                
                # noise window
                noise_start_bf = 1.0 
                noise_pad = 0.3
                noise_start_sample = int((p_arrival - starttime - noise_start_bf)*sampling_rate)
                noise_end_sample  =  int((p_arrival - starttime - noise_pad)*sampling_rate)
                
                # calculate noise mean and std
                noise_E = Tr_E.data[noise_start_sample:noise_end_sample]
                noise_N = Tr_N.data[noise_start_sample:noise_end_sample]
                noise_Z = Tr_Z.data[noise_start_sample:noise_end_sample]
                E_mean, E_std = np.mean(noise_E), np.std(noise_E)
                N_mean, N_std = np.mean(noise_N), np.std(noise_N)
                Z_mean, Z_std = np.mean(noise_Z), np.std(noise_Z)
                
                # data window
                E_data = Tr_E.data[s_arr_sample:(coda_end_sample - 60)]
                N_data = Tr_N.data[s_arr_sample:(coda_end_sample - 60)]
                Z_data = Tr_Z.data[p_arr_sample:(s_arr_sample - 40)]

                # calculate the snr
                snr_E = 10*np.log10(np.mean(E_data**2)/np.mean(noise_E**2))
                snr_N = 10*np.log10(np.mean(N_data**2)/np.mean(noise_N**2))
                snr_Z = 10*np.log10(np.mean(Z_data**2)/np.mean(noise_Z**2))
                
                snr_db = np.array([snr_E, snr_N, snr_Z])
                if any(snr < 1.7 for snr in snr_db):
                    continue
                    
                # maniputlate the background noise and make the waveform isolated
                try:
                    #seismogram = convolve(gaussian_noise, kernel, mode='same')
                    Tr_E.data[:noise_end_sample] = convolve(np.random.normal(E_mean, E_std, (noise_end_sample)), kernel, mode = 'same')
                    Tr_E.data[coda_end_sample:] = convolve(np.random.normal(E_mean, E_std, (3000 - coda_end_sample)), kernel, mode = 'same')
                    Tr_N.data[:noise_end_sample] = convolve(np.random.normal(N_mean, N_std, (noise_end_sample)), kernel, mode = 'same')
                    Tr_N.data[coda_end_sample:] = convolve(np.random.normal(N_mean, N_std, (3000 - coda_end_sample)), kernel, mode = 'same')
                    Tr_Z.data[:noise_end_sample] = convolve(np.random.normal(Z_mean, Z_std, (noise_end_sample)), kernel, mode = 'same')
                    Tr_Z.data[coda_end_sample:] = convolve(np.random.normal(Z_mean, Z_std, (3000 - coda_end_sample)), kernel, mode = 'same')
                except Exception:
                    continue

                # do filter, detrend, and demean once again
                for tr in [Tr_E, Tr_N, Tr_Z]:
                    tr.filter('bandpass', freqmin = 0.75, freqmax = 45, zerophase = True)
                    tr.interpolate(sampling_rate = 100)
                    tr.detrend('demean')
                    tr.taper(0.1, 'cosine')
                    
                # stacking the three component data
                try:
                    data = np.column_stack((Tr_E.data, Tr_N.data, Tr_Z.data))
                    if data.shape != (3000,3):
                        continue
                except Exception:
                    continue
                
                # define p_time and s_time in datetime format
                P_time = str(p_arrival.datetime)
                S_time = str(s_arrival.datetime)
                
                # write the hd5f file
                HDFr = h5py.File(output.joinpath(f'{hdf_out_name}'), 'a')
                dsF = HDFr.create_dataset("data/" + trace_name, data.shape, data = data, dtype = np.float32)
                dsF.attrs['p_time'] = P_time
                dsF.attrs['p_arrival_sample'] = p_arr_sample
                dsF.attrs['p_attrs'] = P_attr
                dsF.attrs['s_time'] = S_time
                dsF.attrs['s_arrival_sample'] = s_arr_sample
                dsF.attrs['s_attrs'] = S_attr
                dsF.attrs['coda_end_sample'] = coda_end_sample_arr
                dsF.attrs['chan_E_metadata'] = Tr_E_mtd
                dsF.attrs['chan_N_metadata'] = Tr_N_mtd
                dsF.attrs['chan_Z_metadata'] = Tr_Z_mtd
                dsF.attrs['trace_name'] = trace_name
                HDFr.flush()       
                HDFr.close()
    return None
    
def main ():
    # path to the file
    hypo = Path(r"C:\Users\User\eqt-project\training\hypo_reloc.xlsx")
    picking = Path(r"C:\Users\User\eqt-project\training\catalog_picking.xlsx")
    station = Path(r"C:\Users\User\eqt-project\training\SEML_station.xlsx")
    wave = Path(r"C:\Users\User\eqt-project\training\wave_ai_60s")
    output = Path(r"C:\Users\User\eqt-project\cross-correlation")
    hdf_name = r'xcorr.hdf5'
    
    build(hypo, picking, station, wave, output, hdf_name)
    
    return None

if name == "__main__":
    main()
        