from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from io import BytesIO
from metpy.units import units
from metpy import calc

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
    hght = raw_dict['Altitude (m AGL)'] * units.meter
    tmpc = raw_dict['Temperature (C)'] * units.degC
    dwpc = calc.dewpoint_from_relative_humidity(raw_dict['Temperature (C)'] * units.degC, raw_dict['Relative humidity (%)'] * units.percent)
    wdir = raw_dict['Heading (degrees)'] * units.degree
    wspd = raw_dict['Speed (m/s)'] * units.meter/units.second
    rise = raw_dict['Rise speed (m/s)'] * units.meter/units.second

    #convert wind to u,v
    wind_u, wind_v = calc.wind_components(wspd, wdir)

    with_balloon_profile = {'pres':pres[:split], 'hght':hght[:split],
                            'tmpc':tmpc[:split], 'dwpc':dwpc[:split],
                            'wdir':wdir[:split], 'wspd':wspd[:split],
                            'rise':rise[:split],
                            'wind_u':wind_u[:split], 'wind_v':wind_v[:split]}

    no_balloon_profile = {'pres':pres[split:], 'hght':hght[split:],
                            'tmpc':tmpc[split:], 'dwpc':dwpc[split:],
                            'wdir':wdir[split:], 'wspd':wspd[split:],
                            'rise':rise[split:],
                            'wind_u':wind_u[split:], 'wind_v':wind_v[split:]}

    return with_balloon_profile, no_balloon_profile, {'time':raw_dict["UTC time"][0], 'lat':raw_dict["Latitude"][0], 'lon':raw_dict["Longitude"][0]}



