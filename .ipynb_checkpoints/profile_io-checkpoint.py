from datetime import datetime, timedelta
import zipfile
import os
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd

from io import BytesIO
from metpy.units import units
from metpy import calc

import pyart

def decode_oax(filename):

    ## read in the file
    f = open(filename, 'rb')
    file_data = f.read().decode('utf-8')
    data = np.array([l.strip() for l in file_data.split('\n')])

    ## necessary index points
    title_idx = np.where( data == '%TITLE%')[0][0]
    start_idx = np.where( data == '%RAW%' )[0][0] + 1
    finish_idx = np.where( data == '%END%')[0][0]

    ## create the plot title
    data_header = data[title_idx + 1].split()
    location = data_header[0]
    time = datetime.strptime(data_header[1][:11], '%y%m%d/%H%M')
    if len(data_header) > 2:
        lat, lon = data_header[2].split(',')
        lat = float(lat)
        lon = float(lon)
    else:
        print('Cannot read lat lon')
        lat = 0
        lon = 0

    if time > datetime.utcnow() + timedelta(hours=1): 
        # If the strptime accidently makes the sounding in the future (like with SARS archive)
        # i.e. a 1957 sounding becomes 2057 sounding...ensure that it's a part of the 20th century
        time = datetime.strptime('19' + data_header[1][:11], '%Y%m%d/%H%M')

    ## put it all together for StringIO
    full_data = '\n'.join(data[start_idx : finish_idx][:])
    sound_data = BytesIO( full_data.encode() )

    ## read the data into arrays
    pres, hght, tmpc, dwpc, wdir, wspd = np.genfromtxt( sound_data, delimiter=',', comments="%", unpack=True )

    return {'pres':pres * units.hectopascal, 'hght':hght * units.meter,
            'tmpc':tmpc * units.degC, 'dwpc':dwpc * units.degC,
            'wdir':wdir * units.degree, 'wspd':wspd * units.meter/units.second,
            'location':location, 'time':time, 'lat':lat, 'lon':lon}

def decode_raw_flight_history(filename, location='', split=-1):

    df = pd.read_csv(filename, header=0, skipinitialspace=True)
    raw_dict = df.to_dict(orient='list')


    pres = np.array(raw_dict['Pressure (Pascal)'])/100 * units.hectopascal
    hght = raw_dict['Altitude (m MSL)'] * units.meter
    tmpc = raw_dict['Temperature (C)'] * units.degC
    dwpc = calc.dewpoint_from_relative_humidity(raw_dict['Temperature (C)'] * units.degC, raw_dict['Relative humidity (%)'] * units.percent)
    wdir = raw_dict['Heading (degrees)'] * units.degree
    wspd = raw_dict['Speed (m/s)'] * units.meter/units.second
    rise = raw_dict['Rise speed (m/s)'] * units.meter/units.second

    #convert wind to u,v
    wind_u, wind_v = calc.wind_components(wspd, wdir)

    #calculate seconds from launch
    launch_dt = datetime.strptime(raw_dict["UTC time"][0], '%H:%M:%S')
    profile_seconds = np.zeros_like(wind_u)
    for i, time_str in enumerate(raw_dict["UTC time"]):
        tmp_dt = datetime.strptime(time_str, '%H:%M:%S')
        profile_seconds[i] = (tmp_dt-launch_dt).total_seconds()


    with_balloon_profile = {'pres':pres[:split+1], 'hght':hght[:split+1],
                            'tmpc':tmpc[:split+1], 'dwpc':dwpc[:split+1],
                            'wdir':wdir[:split+1], 'wspd':wspd[:split+1],
                            'rise':rise[:split+1],
                            'wind_u':wind_u[:split+1], 'wind_v':wind_v[:split+1],
                            'time':profile_seconds[:split+1],
                            'lat':np.array(raw_dict["Latitude"][:split+1]), 'lon':np.array(raw_dict["Longitude"][:split+1])}

    no_balloon_profile = {'pres':pres[split:], 'hght':hght[split:],
                            'tmpc':tmpc[split:], 'dwpc':dwpc[split:],
                            'wdir':wdir[split:], 'wspd':wspd[split:],
                            'rise':rise[split:],
                            'wind_u':wind_u[split:], 'wind_v':wind_v[split:],
                            'time':profile_seconds[split:],
                            'lat':np.array(raw_dict["Latitude"][split:]), 'lon':np.array(raw_dict["Longitude"][split:])}

    return with_balloon_profile, no_balloon_profile, {'time':launch_dt, 'lat':raw_dict["Latitude"][0], 'lon':raw_dict["Longitude"][0]}


def get_accessg_profile(dt, request_lat, request_lon):

    accessg_root = '/g/data/wr45/ops_aps3/access-g/1' #access g

    #extract date components
    model_timestep_hr = 6
    run_hour_str = str(round(dt.hour/model_timestep_hr)*model_timestep_hr).zfill(2) + '00'
    if run_hour_str == '2400':
        run_hour_str = '0000'

    #build path
    accessg_folder = '/'.join([accessg_root, datetime.strftime(dt, '%Y%m%d'), run_hour_str, 'an'])
    #build filenames
    temp_ffn = accessg_folder + '/pl/air_temp.nc'
    relh_ffn = accessg_folder + '/pl/relhum.nc'
    uwnd_ffn = accessg_folder + '/pl/wnd_ucmp.nc'
    vwnd_ffn = accessg_folder + '/pl/wnd_vcmp.nc'
    geop_ffn = accessg_folder + '/pl/geop_ht.nc'

    #extract data
    with xr.open_dataset(temp_ffn) as temp_ds:
        temp_profile = temp_ds.air_temp.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[:][0] - 273.15 #units: deg K -> C
    with xr.open_dataset(relh_ffn) as rh_ds:
        rh_profile = rh_ds.relhum.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[:][0] #units: percentage
    with xr.open_dataset(uwnd_ffn) as uwnd_ds:
        uwnd_profile = uwnd_ds.wnd_ucmp.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[:][0] #units: m/s
    with xr.open_dataset(vwnd_ffn) as vwnd_ds:
        vwnd_profile = vwnd_ds.wnd_vcmp.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[:][0] #units: m/s
    with xr.open_dataset(geop_ffn) as geop_ds:
        geopot_profile = geop_ds.geop_ht.sel(lon=request_lon, method='nearest').sel(lat=request_lat, method='nearest').data[:][0] #units: m
        pres_profile = geop_ds.lvl.data[:]/100 #units: Pa #units: hpa

    temp_profile = np.flipud(temp_profile)
    rh_profile = np.flipud(rh_profile)
    uwnd_profile = np.flipud(uwnd_profile)
    vwnd_profile = np.flipud(vwnd_profile)
    geopot_profile = np.flipud(geopot_profile)   
    pres_profile = np.flipud(pres_profile)   

    dwpc_profile = calc.dewpoint_from_relative_humidity(temp_profile * units.degC, rh_profile * units.percent)

    return {'pres':pres_profile * units.hectopascal, 'hght':geopot_profile * units.meter,
            'tmpc':temp_profile * units.degC, 'dwpc':dwpc_profile,
            'wind_u':uwnd_profile * units.meter/units.second, 'wind_v':vwnd_profile * units.meter/units.second}


def load_level1_radar_data(radar_id, start_dt, end_dt, vol_time=300):

    level_1_path = '/g/data/rq0/level_1/odim_pvol'

    #list of dates between start and end
    date_list = []
    for n in range(int ((end_dt - start_dt).days)+1):
        date_list.append(start_dt + timedelta(n))

    #unzip radar data
    radars = []
    dt_list = []
    for date in date_list:
        #build size file
        zip_ffn = f'{level_1_path}/{radar_id}/{date.year}/vol/{radar_id}_{date.strftime("%Y%m%d")}.pvol.zip'
        #open zip
        with zipfile.ZipFile(zip_ffn) as zip_fd:
            #list contents
            zip_filelist = zip_fd.namelist()[1:]
            for zip_item in zip_filelist:
                #for each item, read dt
                parts = zip_item.split('_')
                vol_dt = datetime.strptime(f'{parts[1]}_{parts[2][0:6]}', '%Y%m%d_%H%M%S') + timedelta(seconds=vol_time/2)  #the filename is the volume start time. add half volume time to better capture the mid point time
                #if dt is in target period
                if vol_dt >= start_dt - timedelta(seconds=vol_time/2) and vol_dt<= end_dt + timedelta(seconds=vol_time/2):
                    #append vol_dt
                    dt_list.append(vol_dt) 
                    #append radar data
                    with zip_fd.open(zip_item) as radarfile:
                        radars.append(pyart.aux_io.read_odim_h5(radarfile))

    return radars, dt_list

def load_nhp_radar_data(data_path, vol_time):

    #list files
    radar_ffn_list = sorted(glob(data_path + '/*.h5'))
    #load data
    radars = []
    dt_list = []
    for radar_ffn in radar_ffn_list:
        #load radar
        radars.append(pyart.aux_io.read_odim_h5(radar_ffn))
        #extract vol time
        vol_dt = datetime.strptime(os.path.basename(radar_ffn)[0:13], '%Y%m%d%H_%M')
        #append vol_dt
        dt_list.append(vol_dt - timedelta(seconds=vol_time/2)) #the filename is the volume end time. subtract half volume time to better capture the mid point time
    return radars, dt_list





