from datetime import datetime, timedelta
import zipfile
import os
import math
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd

from io import BytesIO
from metpy.units import units
from metpy import calc

import pyart

def find_nearest_dt_idx(target_dt, dt_list):
    time_diff = []
    for tempdt in dt_list:
        time_diff.append(abs((target_dt-tempdt).total_seconds()))
    return np.argmin(time_diff)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def degrees2meters(degrees, radius=6371000.0):
    """
    Convenience function to convert (great circle) degrees to meters
    assuming a perfectly spherical Earth.

    :type degrees: float
    :param degrees: Distance in (great circle) degrees
    :type radius: float, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in meters as a floating point number.

    .. rubric:: Example

    >>> from obspy.geodetics import degrees2kilometers
    >>> degrees2kilometers(1)
    111.19492664455873
    """
    return degrees * (2.0 * radius * math.pi / 360.0)

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

def decode_raw_flight_history_nosplit(filename, location='', remove_nan_rows=False):

    df = pd.read_csv(filename, header=0, skipinitialspace=True)
    raw_dict = df.to_dict(orient='list')


    pres = np.array(raw_dict['Pressure (Pascal)'])/100 * units.hectopascal
    hght = raw_dict['Altitude (m MSL)'] * units.meter
    tmpc = raw_dict['Temperature (C)'] * units.degC
    dwpc = calc.dewpoint_from_relative_humidity(raw_dict['Temperature (C)'] * units.degC, raw_dict['Relative humidity (%)'] * units.percent)
    wdir = raw_dict['Heading (degrees)'] * units.degree
    wspd = raw_dict['Speed (m/s)'] * units.meter/units.second
    rise = raw_dict['Rise speed (m/s)'] * units.meter/units.second
    utc_time = np.array(raw_dict["UTC time"])
    #convert wind to u,v
    wind_u, wind_v = calc.wind_components(wspd, wdir)

    #get lat/lon
    lat = np.array(raw_dict["Latitude"])
    lon = np.array(raw_dict["Longitude"])

    #calculate seconds from launch
    launch_dt = datetime.strptime(utc_time[0], '%H:%M:%S')
    profile_seconds = np.zeros_like(wind_u)
    for i, time_str in enumerate(utc_time):
        tmp_dt = datetime.strptime(time_str, '%H:%M:%S')
        profile_seconds[i] = (tmp_dt-launch_dt).total_seconds()

    #remove nan rows (occuring in GPS data)
    if remove_nan_rows:
        nan_filter = ~np.isnan(lat)
        pres = pres[nan_filter]
        hght = hght[nan_filter]
        tmpc = tmpc[nan_filter]
        dwpc = dwpc[nan_filter]
        wdir = wdir[nan_filter]
        wspd = wspd[nan_filter]
        rise = rise[nan_filter]
        utc_time = utc_time[nan_filter]
        wind_u = wind_u[nan_filter]
        wind_v = wind_v[nan_filter]
        lat = lat[nan_filter]
        lon = lon[nan_filter]
        profile_seconds = profile_seconds[nan_filter]

    prof = {'pres':pres, 'hght':hght,
                        'tmpc':tmpc, 'dwpc':dwpc,
                        'wdir':wdir, 'wspd':wspd,
                        'rise':rise,
                        'wind_u':wind_u, 'wind_v':wind_v,
                        'time':profile_seconds,
                        'lat':lat, 'lon':lon}



    return prof, {'time':launch_dt, 'lat':lat[0], 'lon':lon[0]}

def decode_raw_flight_history(filename, location='', split=-1, remove_nan_rows=False):

    df = pd.read_csv(filename, header=0, skipinitialspace=True)
    raw_dict = df.to_dict(orient='list')


    pres = np.array(raw_dict['Pressure (Pascal)'])/100 * units.hectopascal
    hght = raw_dict['Altitude (m MSL)'] * units.meter
    tmpc = raw_dict['Temperature (C)'] * units.degC
    dwpc = calc.dewpoint_from_relative_humidity(raw_dict['Temperature (C)'] * units.degC, raw_dict['Relative humidity (%)'] * units.percent)
    wdir = raw_dict['Heading (degrees)'] * units.degree
    wspd = raw_dict['Speed (m/s)'] * units.meter/units.second
    rise = raw_dict['Rise speed (m/s)'] * units.meter/units.second
    utc_time = np.array(raw_dict["UTC time"])
    #convert wind to u,v
    wind_u, wind_v = calc.wind_components(wspd, wdir)

    #get lat/lon
    lat = np.array(raw_dict["Latitude"])
    lon = np.array(raw_dict["Longitude"])



    #calculate seconds from launch
    launch_dt = datetime.strptime(utc_time[0], '%H:%M:%S')
    profile_seconds = np.zeros_like(wind_u)
    for i, time_str in enumerate(utc_time):
        tmp_dt = datetime.strptime(time_str, '%H:%M:%S')
        profile_seconds[i] = (tmp_dt-launch_dt).total_seconds()

    #remove nan rows (occuring in GPS data)
    if remove_nan_rows:
        #split time
        split_time = profile_seconds[split]
        #apply nan filter
        nan_filter = ~np.isnan(lat)
        pres = pres[nan_filter]
        hght = hght[nan_filter]
        tmpc = tmpc[nan_filter]
        dwpc = dwpc[nan_filter]
        wdir = wdir[nan_filter]
        wspd = wspd[nan_filter]
        rise = rise[nan_filter]
        utc_time = utc_time[nan_filter]
        wind_u = wind_u[nan_filter]
        wind_v = wind_v[nan_filter]
        lat = lat[nan_filter]
        lon = lon[nan_filter]
        profile_seconds = profile_seconds[nan_filter]
        #adjust split
        split = np.argmin(np.abs(split_time - profile_seconds))

    with_balloon_profile = {'pres':pres[:split+1], 'hght':hght[:split+1],
                            'tmpc':tmpc[:split+1], 'dwpc':dwpc[:split+1],
                            'wdir':wdir[:split+1], 'wspd':wspd[:split+1],
                            'rise':rise[:split+1],
                            'wind_u':wind_u[:split+1], 'wind_v':wind_v[:split+1],
                            'time':profile_seconds[:split+1],
                            'lat':lat[:split+1], 'lon':lon[:split+1]}

    no_balloon_profile = {'pres':pres[split:], 'hght':hght[split:],
                            'tmpc':tmpc[split:], 'dwpc':dwpc[split:],
                            'wdir':wdir[split:], 'wspd':wspd[split:],
                            'rise':rise[split:],
                            'wind_u':wind_u[split:], 'wind_v':wind_v[split:],
                            'time':profile_seconds[split:],
                            'lat':lat[split:], 'lon':lon[split:]}

    return with_balloon_profile, no_balloon_profile, {'time':launch_dt, 'lat':lat[0], 'lon':lon[0]}


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

    """
    Notes on Canadian radar volumes
    - top down 6min volume
    - file timestamp is the end time (closest to lowest elevation) rounded up to nearest minute
    - the scan time in the files therefore starts from about -6m
    """

    #list files
    radar_ffn_list = sorted(glob(data_path + '/*.h5'))
    #load data
    radars = []
    mid_point_dt_list = []
    file_dt_list = []
    for radar_ffn in radar_ffn_list:
        #load radar
        radars.append(pyart.aux_io.read_odim_h5(radar_ffn, file_field_names=True))
        #extract vol time
        vol_dt = datetime.strptime(os.path.basename(radar_ffn)[0:13], '%Y%m%d%H_%M')
        #append vol_dt
        mid_point_dt_list.append(vol_dt - timedelta(seconds=vol_time/2)) #the filename is the volume end time. subtract half volume time to better capture the mid point time
        file_dt_list.append(vol_dt)
    return radars, file_dt_list, mid_point_dt_list

def load_nhp_phido_radar_data(data_path, vol_time):

    """
    Notes on Canadian radar volumes
    - top down 6min volume
    - file timestamp is the end time (closest to lowest elevation) rounded up to nearest minute
    - the scan time in the files therefore starts from about -6m
    """

    #list files
    radar_ffn_list = sorted(glob(data_path + '/*.nc'))
    #load data
    radars = []
    mid_point_dt_list = []
    file_dt_list = []
    for radar_ffn in radar_ffn_list:
        #load radar
        radars.append(pyart.io.read_cfradial(radar_ffn, file_field_names=True))
        #extract vol time
        vol_dt = datetime.strptime(os.path.basename(radar_ffn)[0:13], '%Y%m%d%H_%M')
        #append vol_dt
        mid_point_dt_list.append(vol_dt - timedelta(seconds=vol_time/2)) #the filename is the volume end time. subtract half volume time to better capture the mid point time
        file_dt_list.append(vol_dt)
    return radars, file_dt_list, mid_point_dt_list



