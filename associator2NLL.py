import os, glob
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from typing import Dict, Tuple, List, Optional


def associate(detec_dir: str, output_dir: str) -> None:
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
        
    # build the df
    df = pd.DataFrame()
    for (parent, child, files) in os.walk(detec_dir):
        for file in files:
            if file.endswith('.csv'):
                file_name = os.path.join(parent, file)
                read_csv = pd.read_csv(file_name, delimiter=',')
                df= pd.concat([df, read_csv], ignore_index = True)
                
    # Filter and process the dataframe
    df = df[df["p_arrival_time"].notna() & df["p_snr"].notna() & df["s_snr"].notna()]
    df = df.query('p_snr >= 1.0 and s_snr >= 1.0')
    df["p_arrival_time"] = df["p_arrival_time"].apply(UTCDateTime)
    df["s_arrival_time"] = df["s_arrival_time"].fillna(UTCDateTime("1970-01-01 12:23:34"))
    df["s_arrival_time"] = df["s_arrival_time"].apply(UTCDateTime)
    df["Ts_Tp"] = df['s_arrival_time'] - df['p_arrival_time']
    df = df.query('Ts_Tp < 3')
    df=df.sort_values(by = ['p_arrival_time'])
    last_df = pd.DataFrame([[-999] * df.shape[1]], columns = df.columns)
    df = pd.concat([df, last_df], ignore_index=True)
    df.index = [i for i in range(0,len(df))]

    # start writing
    holder_pick = list()
    time_r = UTCDateTime("1970-01-01 12:23:34")
    holder = list()
    index = 0
    while True:
        try:
            p_arr = df.at[index, 'p_arrival_time']
            test_format = p_arr - time_r
        except TypeError as e:
            holder_pick.append(holder)
            holder = list()
            break
        if index == 0:
            time_r = p_arr
        if  p_arr - time_r <= 4:
            data = list(df.loc[index,:])
            holder.append(data)
            index+=1
        elif p_arr - time_r >= 4 :
            time_r = p_arr
            holder_pick.append(holder)
            holder = list()
        else:
            pass
    
    pick_counter = 0
    for k in holder_pick:
        if len(k) < 3:
            continue
        pick_counter += 1
        dt = k[0][11]
        name = f"{dt.year}{dt.month:02d}{dt.day} {dt.hour:02d}{dt.minute:02d} {dt.second:02d}.pick"
        print(f"Associating {name}")
        with open(os.path.join(output_dir, name), 'w') as file_write:
            for j in k:
                if j[15] != UTCDateTime("1970-01-01 12:23:34"):
                    p_time, s_time = j[11], j[15]
                    line = (f"{j[2]} ? BHZ i P ? {p_time.year}{p_time.month:02d}{p_time.day:02d} "
                    f"{p_time.hour:02d}{p_time.minute:02d} {(p_time.second + p_time.microsecond/1e6):8.5f} "
                    f"GAU 0.0 0.0 0.0 0.0\n"
                    f"{j[2]} ? BHE e S ? {s_time.year}{s_time.month:02d}{s_time.day:02d} "
                    f"{s_time.hour:02d}{s_time.minute:02d} {(s_time.second + s_time.microsecond/1e6):8.5f} "
                    f"GAU 0.0 0.0 0.0 0.0\n")
                else:
                    p_time = j[11]
                    line =( f"{j[2]} ? BHZ i P ? {p_time.year}{p_time.month:02d}{p_time.day:02d} "
                    f"{p_time.hour:02d}{p_time.minute:02d} {(p_time.second + p_time.microsecond/1e6):8.5f} "
                    f"GAU 0.0 0.0 0.0 0.0\n")
                file_write.write(line)
    print(f"There are {pick_counter} picks succesfully associated - - - - -", flush = True)
    return None

if __name__ == "__main__":
    detec_dir = r"C:\Users\User\eqt-project\detections3"
    output_dir = r"C:\Users\User\eqt-project\picks"
    associate(detec_dir, output_dir)