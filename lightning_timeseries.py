import numpy as np
import boto3
import botocore
from botocore.client import Config
import io
import pandas as pd
import matplotlib.pyplot as plt
from ftplib import FTP
import cartopy.crs as ccrs
import datetime as dt
from netCDF4 import Dataset
from io import BytesIO
from scipy import stats
import matplotlib.dates as mdates
#import os

def flatten(t):
    #removes nested lists
    return [item for sublist in t for item in sublist]

def get_doy(year, month, day):
    return dt.datetime(year, month, day).timetuple().tm_yday

def group_filepaths(filepaths):

    start_datetime = []
    filepaths_grouped = []

    for i in range(0, len(filepaths), 15):
        filepaths_grouped.append(filepaths[i:i+15])

        #Obtain year, month, day, hour, and minute from filename after s
        year = int(filepaths[i].split('/')[1])
        day = int(filepaths[i].split('/')[2])
        hour = int(filepaths[i].split('/')[3])
        minute = int(filepaths[i].split('/')[4][29:31])

        #Create datetime object
        start_datetime.append(dt.datetime(year, 1, 1) + dt.timedelta(day - 1) + dt.timedelta(hours = hour) + dt.timedelta(minutes = minute))

    return start_datetime, filepaths_grouped

def realtime_read_glm_filelist(filelist, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, apply_qc=True,output_qc=False):

    s3 = boto3.resource('s3', config = Config(signature_version = botocore.UNSIGNED, user_agent_extra = 'Resource'))
    bucket = s3.Bucket('noaa-goes16')

    '''
    This function will read in the filelist of glm netcdf files and then output arrays of the lightning groups
    '''

    #For Groups
    lats_l = []
    lons_l = []
    ener_l = []
    area_l = []
    flag_l = []    
    flid_l = []

    #For Flashes
    lats_lf = []
    lons_lf = []
    ener_lf = []
    area_lf = []
    flag_lf = []   
    flnum_lf = []

    if len(filelist)<1:
        return np.array(lats_l), np.array(lons_l), np.array(ener_l), np.array(area_l), np.array(flag_l), np.array(flid_l), np.array(lats_lf), np.array(lons_lf), np.array(ener_lf), np.array(area_lf), np.array(flag_lf)

    for filename in filelist:
    	
        #filestats = os.stat(os.path.join(filepath,filename))

        if filename == None:
            break
            
        #elif filestats.st_size == 0 or filename == 'OR_GLM-L2-LCFA_G16_s20232362201200_e20232362201400_c20232362201416.nc' or filename == 'OR_GLM-L2-LCFA_G16_s20232362201000_e20232362201200_c20232362201215.nc' or filename == 'OR_GLM-L2-LCFA_G16_s20232362202000_e20232362202200_c20232362202217.nc' or filename == 'OR_GLM-L2-LCFA_G16_s20232362202400_e20232362203000_c20232362203018.nc':
         #   continue

        #GLM_file = Dataset(os.path.join(filepath, filename))
        obj = s3.Object('noaa-goes16', filename)
        response = obj.get()
        data = response['Body'].read()
        GLM_file = Dataset('file', memory = BytesIO(data).getvalue())

        #Area units were changed in 2019 from km2 to m2 so convert to m2
        area_unit = GLM_file.variables['group_area'].units
        
        if area_unit=='m2':
            area_conv = 1.
        elif area_unit=='km2':
            area_conv=1000.**2

        temp_lats_g = GLM_file.variables['group_lat'][:]
        temp_lons_g = GLM_file.variables['group_lon'][:] + 360. #add 360 to convert to similar format as other grids
        temp_flag_g = GLM_file.variables['group_quality_flag'][:]

        temp_lats_f = GLM_file.variables['flash_lat'][:]
        temp_lons_f = GLM_file.variables['flash_lon'][:] + 360. #add 360 to convert to similar format as other grids
        temp_flag_f = GLM_file.variables['flash_quality_flag'][:]


        cond_g = np.squeeze([((temp_flag_g==0)&(temp_lats_g<=LAT_MAX)&(temp_lats_g>=LAT_MIN)&(temp_lons_g<=LON_MAX)&(temp_lons_g>=LON_MIN))])
        cond_f = np.squeeze([((temp_flag_f==0)&(temp_lats_f<=LAT_MAX)&(temp_lats_f>=LAT_MIN)&(temp_lons_f<=LON_MAX)&(temp_lons_f>=LON_MIN))])

        #For the group quantities
        lats_l.append(list(temp_lats_g[cond_g]))
        lons_l.append(list(temp_lons_g[cond_g]))
        ener_l.append(list(GLM_file.variables['group_energy'][:][cond_g]))
        area_l.append(list(area_conv*GLM_file.variables['group_area'][:][cond_g]))
        flag_l.append(list(temp_flag_g[cond_g]))
        flid_l.append(list(GLM_file.variables['group_parent_flash_id'][:][cond_g]))

        #For flash information
        lats_lf.append(list(temp_lats_f[cond_f]))
        lons_lf.append(list(temp_lons_f[cond_f]))   
        ener_lf.append(list(GLM_file.variables['flash_energy'][:][cond_f]))
        area_lf.append(list(area_conv*GLM_file.variables['flash_area'][:][cond_f]))
        flag_lf.append(list(temp_flag_f[cond_f]))
        flnum_lf.append(list(GLM_file.variables['flash_id'][:][cond_f]))


        GLM_file.close()
    
    #Reformat the group variables as numpy arrays
    lats_l = np.array(flatten(lats_l))
    lons_l = np.array(flatten(lons_l)) 
    ener_l = np.array(flatten(ener_l))
    area_l = np.array(flatten(area_l))
    flag_l = np.array(flatten(flag_l))
    flid_l = np.array(flatten(flid_l))

    #Reformat the flash variables as numpy arrays
    lats_lf = np.array(flatten(lats_lf))
    lons_lf = np.array(flatten(lons_lf)) 
    ener_lf = np.array(flatten(ener_lf))
    area_lf = np.array(flatten(area_lf))
    flag_lf = np.array(flatten(flag_lf))
    flnum_lf = np.array(flatten(flnum_lf))

    #if we need to qc the data
    if apply_qc==True:
        #Defined Exponential Function to eliminate questionable data
        AA = 0.9266284080291266
        BB = 2.1140353739008016e+20
        
        #Using the exponential, given the energy, what should the threshhold be for area
        Area_thresh = 10.**(AA*np.log10(ener_l)+np.log10(BB))
        Area_threshF = 10.**(AA*np.log10(ener_lf)+np.log10(BB))
        
        #Energy minimum to use as cutoff
        Energy_min = 1.9*1e-15
        
        #These are the QC conditions, above the area thresh and energy min, flag must be zero, and exclude non physical lat/lons
        condition_not = np.squeeze([((area_l<Area_thresh)|(ener_l<Energy_min))])
#        condition_ = np.squeeze([((area_l>=Area_thresh)&(ener_l>=Energy_min)&(flag_l==0)&(lats_l<=CONFIG.LAT_MAX)&(lats_l>=CONFIG.LAT_MIN)&(lons_l<=CONFIG.LON_MAX)&(lons_l>=CONFIG.LON_MIN))])

         #Get the unique flash id for all bad flashes
        bad_flash = flid_l[condition_not]

        # get the condition for only non bad identified groups using flash id
        condition_ = np.squeeze([(flid_l!=bad_flash)])

        condition_f = np.array([i for i,val in enumerate(flnum_lf) if val not in np.unique(bad_flash)]) 
        # now condition the flashes using bad flash ids
#        condition_f = np.squeeze([(flnum_lf==valid_flash)])

#        condition_f = np.squeeze([((area_lf>=Area_threshF)&(ener_lf>=Energy_min)&(flag_lf==0)&(lats_lf<=CONFIG.LAT_MAX)&(lats_lf>=CONFIG.LAT_MIN)&(lons_lf<=CONFIG.LON_MAX)&(lons_lf>=CONFIG.LON_MIN))])

        #Uncomment the line below to not include the QC equation
#        condition_ = np.squeeze([(flag_l==0)&(lats_l<=CONFIG.LAT_MAX)&(lats_l>=CONFIG.LAT_MIN)&(lons_l<=CONFIG.LON_MAX)&(lons_l>=CONFIG.LON_MIN)])

        lats_ = np.array(flatten(lats_l[condition_]))
        lons_ = np.array(flatten(lons_l[condition_]))
        ener_ = np.array(flatten(ener_l[condition_]))
        area_ = np.array(flatten(area_l[condition_]))
        flag_ = np.array(flatten(flag_l[condition_]))    
        flid_ = np.array(flatten(flid_l[condition_]))    

        lats_f = np.array((lats_lf[condition_f]))
        lons_f = np.array((lons_lf[condition_f])) 
        ener_f = np.array((ener_lf[condition_f]))
        area_f = np.array((area_lf[condition_f]))
        flag_f = np.array((flag_lf[condition_f]))

     
        return lats_, lons_, ener_, area_, flag_, flid_, lats_f, lons_f, ener_f, area_f, flag_f

    else: #If no QC is applied to the lightning data

        return lats_l, lons_l, ener_l, area_l, flag_l, flid_l, lats_lf, lons_lf, ener_lf, area_lf, flag_lf

def bin_statistic(lat,lon,val,lat_b,lon_b,STAT_TYPE):
    #Need
    #from scipy import stats
    #This function will bin ungridded data to the desired grid and calculate a given statistic within the bins
    #STAT_TYPE = 'count' 'mean' 'std' 'min' 'max'

    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(
        lon, lat, values=val, statistic=STAT_TYPE, bins=[lon_b, lat_b])

    return statistic



current_dt = dt.datetime.now()
YYYY = current_dt.year
DOY = get_doy(current_dt.year, current_dt.month, current_dt.day)

LON_MIN = 275
LON_MAX = 290
LAT_MIN = 10.0
LAT_MAX = 15.0

LAT_ARRAY = np.arange(LAT_MIN, LAT_MAX + 0.02, 0.02)
LON_ARRAY = np.arange(LON_MIN, LON_MAX + 0.02, 0.02)

#Create bins which extend beyond the lat min/max
LAT_BINS = np.append(LAT_ARRAY,(LAT_ARRAY[-1]+0.02))
LAT_BINS = LAT_BINS-0.02/2.

#Create bins which extend beyond the lon min/max
LON_BINS = np.append(LON_ARRAY,(LON_ARRAY[-1]+0.02))
LON_BINS = LON_BINS-0.02/2.

LON_M_ARRAY, LAT_M_ARRAY  = np.meshgrid(LON_ARRAY, LAT_ARRAY)

s3 = boto3.resource('s3', config = Config(signature_version = botocore.UNSIGNED, user_agent_extra = 'Resource'))
bucket = s3.Bucket('noaa-goes16')

filepaths = []
for file in bucket.objects.filter(Prefix = f'GLM-L2-LCFA/{YYYY}/{DOY}'):
    filepaths.append(file.key)

start_datetime, filepaths_grouped = group_filepaths(filepaths)

total_energy_list = []
mean_group_area_list = []
group_count_list = []

for idx, filegroup in enumerate(filepaths_grouped):

    print(f"Processing file group {idx+1} of {len(filepaths_grouped)}")

    lats_l, lons_l, ener_l, area_l, flag_l, flid_l, lats_lf, lons_lf, ener_lf, area_lf, flag_lf = realtime_read_glm_filelist(filepaths_grouped[idx], LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, apply_qc=False, output_qc=False)

    total_energy = np.transpose(bin_statistic(lats_l[~np.isnan(ener_l)],lons_l[~np.isnan(ener_l)],ener_l[~np.isnan(ener_l)],LAT_BINS,LON_BINS,'sum').astype(np.float32))
    mean_group_area = np.transpose(bin_statistic(lats_l[~np.isnan(ener_l)],lons_l[~np.isnan(ener_l)],ener_l[~np.isnan(ener_l)],LAT_BINS,LON_BINS,'sum').astype(np.float32))
    #group_count = np.transpose(bin_statistic(lats_l,lons_l,ener_l,LAT_BINS,LON_BINS,'count'))

    total_energy_list.append(np.sum(total_energy))
    #mean_group_area_list.append(mean_group_area)
    #group_count_list.append(group_count)


# Plot the total energy
fig = plt.figure(figsize=(11,8.5))
ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
ax.coastlines(resolution='50m')

# Add gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

lng_eng_masked = np.ma.masked_where(total_energy == 0, total_energy)
lng_area_masked = np.ma.masked_where(mean_group_area == 0, mean_group_area)

#Plot lightning energy and area using weighted scatterplot
flattened_energy = lng_eng_masked.flatten()
flattened_area = lng_area_masked.flatten()
flattened_lons = LON_M_ARRAY.flatten()
flattened_lats = LAT_M_ARRAY.flatten()

combined = np.array(list(zip(flattened_lons, flattened_lats, flattened_energy, flattened_area)),
            dtype=[('longitude', 'f4'), ('latitude', 'f4'), ('energy', 'f4'), ('area', 'f4')])


sorted_combined = np.sort(combined, order='energy')

# Extract the sorted longitude, latitude, and energy
sorted_longitude = sorted_combined['longitude']
sorted_latitude = sorted_combined['latitude']
sorted_energy = sorted_combined['energy']
sorted_area = sorted_combined['area']

# Scale the sorted energy values for visualization
scaled_sorted_energy = sorted_energy * 1e13

# Scale the sorted area values for visualization
scaled_sorted_area = sorted_area

# Plot with the sorted values
scatter = ax.scatter(sorted_longitude, sorted_latitude, s=scaled_sorted_area, c=scaled_sorted_energy, cmap='viridis', transform=ccrs.PlateCarree(), alpha=0.6, marker='o', vmin = 0, vmax = 10)

fig.colorbar(scatter, orientation = 'vertical', label = 'Total Energy * 1e13 (J)', pad = 0.1, shrink = 0.6)
plt.title('5 Min GLM Total Energy')
plt.tight_layout(pad = 1.0)
plt.savefig('map.png')
plt.close()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(start_datetime, total_energy_list)
ax.plot(start_datetime[-1], total_energy_list[-1], marker = '*')
plt.title(f'Total Energy Time Series within Domain of {LAT_MIN} to {LAT_MAX} Latitude and {LON_MIN - 360 } to {LON_MAX - 360} Longitude') 
ax.set_xlabel('Time')
ax.set_ylabel('Total Energy (J)')

# Format the x-axis for dates to include month day and time (label formatting, rotation)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.autofmt_xdate()
plt.savefig('timeseries.png')

plt.close()
