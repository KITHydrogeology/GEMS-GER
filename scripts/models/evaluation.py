# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:12:05 2025

@author: Tanja Liesch
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


pth_dt_dyn = "./data/dynamic"
path_results_single = "./results_single"
path_results_glob_dynonly = "./results_dynonly/consolidated_obs_sim/"
path_results_glob_dynstat = "./results_dynstat/consolidated_obs_sim/"

    

#%% Evaluate Single Well Models

# # Read data for computing mean obs before test set for NSE
date_start_test = pd.to_datetime("2013-01-01", format = "%Y-%m-%d")


#-----------------------------------

# list dynamic time series data files
dt_list_files = os.listdir(pth_dt_dyn)
temp = [i for i,sublist in enumerate(dt_list_files) if '.csv' in sublist]
dt_list_files = [dt_list_files[i] for i in temp]
del temp

# load dynamic time series
dt_list_dyn = list()
for i in range(len(dt_list_files)):
    temp = pd.read_csv(pth_dt_dyn + "/" + dt_list_files[i], 
                       parse_dates=[0], index_col=0, dayfirst = True, decimal = '.', sep=',')
    dt_list_dyn.append(temp)
del temp

# get ID names
dt_list_names = [item[:-4] for item in dt_list_files]

#-----------------------------------


def get_gwl_series(ID, dt_list_names, dt_list_dyn):
    """
    Returns the GWL time series for a given ID.
    """
    try:
        idx = dt_list_names.index(ID)
        
    except ValueError:
        raise ValueError(f"ID '{ID}' not found in dt_list_names.")
    
    df = dt_list_dyn[idx]

    if 'GWL' not in df.columns:
        raise KeyError(f"GWL column not found in data for ID '{ID}'.")

    return df['GWL']


all_scores_list = []

ID_list = [item[:-4] for item in dt_list_files]

def import_results(ID, path_result):
       
    res = pd.read_csv(path_results_single+"/"+ID+'_obs_sim.csv', 
                          parse_dates=[0],index_col=0, dayfirst=False,
                          decimal = '.', sep=';')
    
    obs = res["GWL"]
    sim = res.drop(columns=["GWL"])
    test_sim_median = np.median(sim,axis = 1)


    # median scores
    sim = np.asarray(test_sim_median.reshape(-1,1))
    obs = obs.to_numpy()
    obs = np.asarray(obs.reshape(-1,1))
    

    err = sim-obs
    
    
    # Filter the data to get rows before date_start_test
    gwl = get_gwl_series(ID, dt_list_names, dt_list_dyn)
    gwl.index = pd.to_datetime(gwl.index)
    data_before_start = gwl[gwl.index < date_start_test]
    mean_column = np.mean(data_before_start, axis=0)
    
    #err_nash = obs - np.mean(obs)
    err_nash = obs - mean_column
    r = stats.linregress(sim[:,0], obs[:,0]) 

    NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
    R2 = r.rvalue ** 2
    RMSE =  np.sqrt(np.mean(err ** 2))
    Bias = np.mean(err)
    
    return ID, NSE, R2, RMSE, Bias
    


for ID in ID_list:
    file_path = os.path.join(path_results_single, f"{ID}_obs_sim.csv")
    if not os.path.exists(file_path):
        print(f"File not found for ID {ID}, skipping.")
        continue
    ID, NSE, R2, RMSE, Bias = import_results(ID, path_results_single) 
    all_scores_list.append({'ID': ID, 'NSE': NSE, 'R2': R2, 'RMSE': RMSE, 'Bias': Bias})



all_scores = pd.DataFrame(all_scores_list).set_index('ID')
all_scores.to_csv('./results_single/all_median_test_scores.csv', sep=';', decimal='.')


#%% Evaluate Dynonly Model

# Consolidate run sims per ID


# Base path
base_path = "./results_dynonly/"
run_folders = sorted([f for f in os.listdir(base_path) if f.startswith("run")])
output_path = os.path.join(base_path, "consolidated_obs_sim/")
os.makedirs(output_path, exist_ok=True)


# list dynamic time series data files
dt_list_files = os.listdir(pth_dt_dyn)
temp = [i for i,sublist in enumerate(dt_list_files) if '.csv' in sublist]
dt_list_files = [dt_list_files[i] for i in temp]
del temp


all_scores_list = []

ID_list = [item[:-4] for item in dt_list_files]

def import_results(ID, path_result):
       
    res = pd.read_csv(path_results_glob_dynonly+"/"+ID+'_obs_sim.csv', 
                          parse_dates=[0],index_col=0, dayfirst=False,
                          decimal = '.', sep=';')
    
    obs = res["GWL"]
    sim = res.drop(columns=["GWL"])
    test_sim_median = np.median(sim,axis = 1)


    # median scores
    sim = np.asarray(test_sim_median.reshape(-1,1))
    obs = obs.to_numpy()
    obs = np.asarray(obs.reshape(-1,1))
    

    err = sim-obs
    
    
    # Filter the data to get rows before date_start_test
    gwl = get_gwl_series(ID, dt_list_names, dt_list_dyn)
    gwl.index = pd.to_datetime(gwl.index)
    data_before_start = gwl[gwl.index < date_start_test]
    # Compute the mean of the column corresponding to the ID (make sure to use the right column)
    mean_column = np.mean(data_before_start, axis=0)
    
    #err_nash = obs - np.mean(obs)
    err_nash = obs - mean_column
    r = stats.linregress(sim[:,0], obs[:,0]) 

    NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
    R2 = r.rvalue ** 2
    RMSE =  np.sqrt(np.mean(err ** 2))
    Bias = np.mean(err)
    
    return ID, NSE, R2, RMSE, Bias
    

# Now analyse consolidated data

all_scores_list = []

for ID in ID_list:
    file_path = os.path.join(path_results_glob_dynonly, f"{ID}_obs_sim.csv")
    if not os.path.exists(file_path):
        print(f"File not found for ID {ID}, skipping.")
        continue
    ID, NSE, R2, RMSE, Bias = import_results(ID, path_results_glob_dynonly) 
    all_scores_list.append({'ID': ID, 'NSE': NSE, 'R2': R2, 'RMSE': RMSE, 'Bias': Bias})




all_scores = pd.DataFrame(all_scores_list).set_index('ID')
all_scores.to_csv('./results_dynonly/scores_of_median_sim_per_ID.csv', sep=';', decimal='.')


#%% Evaluate Dynstat Model

# # Consolidate run sims per ID

# Base path
base_path = "./results_dynstat/"
run_folders = sorted([f for f in os.listdir(base_path) if f.startswith("run")])
output_path = os.path.join(base_path, "consolidated_obs_sim/")
os.makedirs(output_path, exist_ok=True)

# Use first run to get IDs
for run in run_folders:
    run_path = os.path.join(base_path, run, "results.csv")
    if os.path.exists(run_path):
        df_sample = pd.read_csv(run_path, sep=';' if ';' in open(run_path).readline() else ',')
        break

# Extract list of IDs from columns ending in '_sim'
sim_cols = [col for col in df_sample.columns if col.endswith('_sim')]
all_ids = [col.rsplit('_', 1)[0] for col in sim_cols]

# Loop over IDs and build one file per ID
for ID in all_ids:
    print(ID)
    obs_series = None
    date_index = None
    sim_dict = {}  # run_number: sim_series

    for i, run in enumerate(run_folders):
        run_path = os.path.join(base_path, run, "results.csv")
        if not os.path.exists(run_path):
            continue

        df = pd.read_csv(run_path, sep=';' if ';' in open(run_path).readline() else ',')

        sim_col = f"{ID}_sim"
        obs_col = f"{ID}_obs"
        if sim_col not in df.columns or obs_col not in df.columns:
            continue

        sim_series = df[sim_col].copy()
        sim_dict[str(i)] = sim_series  # run index as string column name

        if obs_series is None:
            obs_series = df[obs_col].copy()
            date_index = df.index if df.index.name else pd.RangeIndex(start=0, stop=len(obs_series))

    if obs_series is None or not sim_dict:
        continue  # skip this ID if no data found

    # Combine into one DataFrame
    df_out = pd.DataFrame(sim_dict)
    df_out.insert(0, "GWL", obs_series)
    df_out.insert(0, "Date", date_index)

    # Save to CSV
    file_name = f"{ID}_obs_sim.csv"
    df_out.to_csv(os.path.join(output_path, file_name), sep=';', index=False, decimal='.')

# Now analyse consolidated data

def import_results(ID, path_result):
       
    res = pd.read_csv(path_results_glob_dynstat+"/"+ID+'_obs_sim.csv', 
                          parse_dates=[0],index_col=0, dayfirst=False,
                          decimal = '.', sep=';')
    
    obs = res["GWL"]
    sim = res.drop(columns=["GWL"])
    test_sim_median = np.median(sim,axis = 1)


    # median scores
    sim = np.asarray(test_sim_median.reshape(-1,1))
    obs = obs.to_numpy()
    obs = np.asarray(obs.reshape(-1,1))
    

    err = sim-obs
    
    
    # Filter the data to get rows before date_start_test
    gwl = get_gwl_series(ID, dt_list_names, dt_list_dyn)
    gwl.index = pd.to_datetime(gwl.index)
    data_before_start = gwl[gwl.index < date_start_test]
    mean_column = np.mean(data_before_start, axis=0)
    
    #err_nash = obs - np.mean(obs)
    err_nash = obs - mean_column
    r = stats.linregress(sim[:,0], obs[:,0]) 

    NSE = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2)))
    R2 = r.rvalue ** 2
    RMSE =  np.sqrt(np.mean(err ** 2))
    Bias = np.mean(err)
    
    return ID, NSE, R2, RMSE, Bias
    

all_scores_list = []

for ID in ID_list:
    file_path = os.path.join(path_results_glob_dynstat, f"{ID}_obs_sim.csv")
    if not os.path.exists(file_path):
        print(f"File not found for ID {ID}, skipping.")
        continue
    ID, NSE, R2, RMSE, Bias = import_results(ID, path_results_glob_dynstat) 
    all_scores_list.append({'ID': ID, 'NSE': NSE, 'R2': R2, 'RMSE': RMSE, 'Bias': Bias})



all_scores = pd.DataFrame(all_scores_list).set_index('ID')
all_scores.to_csv('./results_dynstat/scores_of_median_sim_per_ID.csv', sep=';', decimal='.')


#%% Merge Scores

mean_scores_single = pd.read_csv("./results_single/all_median_test_scores.csv", 
                      index_col=0, decimal = '.', sep=';')

mean_scores_glob_dynonly = pd.read_csv("./results_dynonly/scores_of_median_sim_per_ID.csv", 
                      index_col=0, decimal = '.', sep=';')

mean_scores_glob_dynstat = pd.read_csv("C./results_dynstat/scores_of_median_sim_per_ID.csv", 
                      index_col=0, decimal = '.', sep=';')

# Rename columns
mean_scores_single = mean_scores_single.rename(columns=lambda x: f"{x}_single")
mean_scores_glob_dynonly = mean_scores_glob_dynonly.rename(columns=lambda x: f"{x}_glob_dynonly")
mean_scores_glob_dynstat = mean_scores_glob_dynstat.rename(columns=lambda x: f"{x}_glob_dynstat")

# Join on ID (index)
merged_scores = mean_scores_single.join(mean_scores_glob_dynonly, how='outer') \
                                  .join(mean_scores_glob_dynstat, how='outer')

# Optional: sort by ID
#merged_scores = merged_scores.sort_index()

# Save if needed
# merged_scores.to_csv("./merged_scores_comparison.csv", sep=';', decimal=',')

#%% Numerical evaluation

nse_data = merged_scores[['NSE_single', 'NSE_glob_dynonly', 'NSE_glob_dynstat']]
nse_data.columns = ['Single', 'Global Dyn Only', 'Global Dyn+Stat']

# Compute summary statistics
nse_summary = pd.DataFrame({
    'Mean NSE': nse_data.mean(),
    'Median NSE': nse_data.median(),
    'Min NSE': nse_data.min(),
    'Max NSE': nse_data.max()
})

print(nse_summary)

r2_data = merged_scores[['R2_single', 'R2_glob_dynonly', 'R2_glob_dynstat']]
r2_data.columns = ['Single', 'Global Dyn Only', 'Global Dyn+Stat']

# Compute summary statistics
r2_summary = pd.DataFrame({
    'Mean R2': r2_data.mean(),
    'Median R2': r2_data.median(),
    'Min R2': r2_data.min(),
    'Max R2': r2_data.max()
})

print(r2_summary)

rmse_data = merged_scores[['RMSE_single', 'RMSE_glob_dynonly', 'RMSE_glob_dynstat']]
rmse_data.columns = ['Single', 'Global Dyn Only', 'Global Dyn+Stat']

# Compute summary statistics
rmse_summary = pd.DataFrame({
    'Mean RMSE': rmse_data.mean(),
    'Median RMSE': rmse_data.median(),
    'Min RMSE': rmse_data.min(),
    'Max RMSE': rmse_data.max()
})

print(rmse_summary)

bias_data = merged_scores[['Bias_single', 'Bias_glob_dynonly', 'Bias_glob_dynstat']]
bias_data.columns = ['Single', 'Global Dyn Only', 'Global Dyn+Stat']

# Compute summary statistics
bias_summary = pd.DataFrame({
    'Mean Bias': bias_data.mean(),
    'Median Bias': bias_data.median(),
    'Min Bias': bias_data.min(),
    'Max Bias': bias_data.max()
})

print(bias_summary)

#%%  Count results

count_results = {}

columns = ['NSE_single', 'NSE_glob_dynonly', 'NSE_glob_dynstat']

# Loop through each column to count the values based on the specified conditions
for column in columns:
    count_results[column] = {
        'below_0': (merged_scores[column] <= 0).sum(),  # Count values < 0
        '0 to 0.5': ((merged_scores[column] > 0) & (merged_scores[column] <= 0.5)).sum(),  # Count values > 0 and <= 0.5
        '0.5-0.8': ((merged_scores[column] > 0.5) & (merged_scores[column] <= 0.8)).sum(),  # Count values > 0.5 and <= 0.8
        'greater_than_0.8': (merged_scores[column] > 0.8).sum()  # Count values > 0.8
    }

# Convert the results to a DataFrame for easy viewing
count_df = pd.DataFrame(count_results)

# Display the results
print(count_df)

#%% Further numerical evaluation

merged_scores_compared = merged_scores.copy()
merged_scores_compared['NSE dynstat - single'] = merged_scores_compared['NSE_glob_dynstat'] - merged_scores_compared['NSE_single']

columns_to_keep = ['NSE_single', 'NSE_glob_dynstat', 'NSE dynstat - single']
# Create a new DataFrame with only the columns you want to keep
temp = merged_scores_compared[columns_to_keep]
merged_scores_compared = temp.copy()

# Count MW under value x in all nse_data
count_negative_rows = len(nse_data[(nse_data.iloc[:, 0] < 0) & (nse_data.iloc[:, 1] < 0) & (nse_data.iloc[:, 2] < 0)])

print(count_negative_rows)

# Count MW under value x in all nse_data
count_not_satisfactory_rows = len(nse_data[(nse_data.iloc[:, 0] < 0.5) & (nse_data.iloc[:, 1] < 0.5) & (nse_data.iloc[:, 2] < 0.5)])

print(count_not_satisfactory_rows)

# Count MW under value x in all nse_data
count_acceptable_rows = len(nse_data[(nse_data.iloc[:, 0] > 0.5) & (nse_data.iloc[:, 1] > 0.5) & (nse_data.iloc[:, 2] > 0.5)])

print(count_acceptable_rows)



#%% Combined Boxplot and CDF



# Prepare and rename columns
nse_data = merged_scores[['NSE_single', 'NSE_glob_dynonly', 'NSE_glob_dynstat']]
nse_data.columns = ['Single', 'Global Dyn Only', 'Global Dyn+Stat']

# Set colors
box_colors = ['salmon', 'lightsteelblue', 'steelblue']

# Create a figure with 1 row and 2 columns (side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figsize as needed

# -----------------------------------
# Boxplot on ax1
# -----------------------------------
# Create boxplot
box = ax1.boxplot(nse_data.values, patch_artist=True, widths=0.4)

# Color and style each box
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')

# Set black edges for whiskers, caps, medians
for element in ['whiskers', 'caps', 'medians']:
    for line in box[element]:
        line.set_color('black')

# Set labels and limits for boxplot
ax1.set_xticklabels(nse_data.columns)
ax1.set_ylabel('Nash–Sutcliffe Efficiency (NSE)')
ax1.set_ylim(-2, 1)
ax1.set_title('Comparison of NSE Scores Across Models')
ax1.grid(True, linestyle='--', alpha=0.5)

# -----------------------------------
# CDF plot on ax2
# -----------------------------------
# Define a list of colors for each column
colors = ['salmon', 'lightsteelblue', 'steelblue']

# Create CDF plot
for i, col in enumerate(nse_data.columns):
    sorted_vals = np.sort(nse_data[col])
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf, label=col, color=colors[i % len(colors)])

# Styling for CDF plot
ax2.set_xlabel('Nash–Sutcliffe Efficiency (NSE)')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('CDF of NSE Scores Across Models')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
ax2.set_xlim(-1, 1)  # Optional: match range to boxplot

# Adjust layout to make sure everything fits well
plt.tight_layout()

plt.subplots_adjust(wspace=0.18)  # Adjust space between subplots

# # Export to PDF
plt.tight_layout()
plt.savefig("nse_boxplot_cdf.pdf", format='pdf')  # Save as PDF

# Show the combined plot
plt.show()



#%% Plot Maps

import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import pyproj
import matplotlib.pyplot as plt
import numpy as np

# Load Well list
temp = pd.read_csv('./data/static/metadata_static.csv',sep=',',decimal='.') # All locations that should be processed
# List of columns to keep
columns_to_keep = ['MW_ID', 'Easting (EPSG:3035)', 'Northing (EPSG:3035)']
# Create a new DataFrame with only the columns you want to keep
well_list = temp[columns_to_keep]
well_list.set_index('MW_ID', inplace=True)

# Define Projections
proj_etrs = pyproj.Proj(init="epsg:25833 ")
proj_wgs84 = pyproj.Proj(init="epsg:4326")
proj_utm = pyproj.Proj(init="epsg:25832")
proj_data = pyproj.Proj(init="epsg:3035")

# Load Bundesländer shapefile
bundeslaender = gpd.read_file("./data/GIS/LAN_ew_22.shp")
bundeslaender2 = gpd.read_file("./data/GIS/VG250_LAN.shp")


# Ensure CRS matches Web Mercator (EPSG:3857)
bundeslaender = bundeslaender.to_crs(epsg=3035)
bundeslaender2 = bundeslaender2.to_crs(epsg=3035)
bundeslaender2 = bundeslaender2[(bundeslaender2['GF'] != 1) & (bundeslaender2['GF'] != 2)]


# Merge well_list and merged_scores
temp = well_list.join(merged_scores, how='inner')
temp2 = merged_scores_compared.drop(columns=['NSE_single', 'NSE_glob_dynstat'])
well_list_scores = temp.join(temp2, how='inner')


# Create geometry from lat/lon
geometry = [Point(xy) for xy in zip(well_list_scores['Easting (EPSG:3035)'], well_list_scores['Northing (EPSG:3035)'])]
gdf = gpd.GeoDataFrame(well_list_scores, geometry=geometry, crs="EPSG:3035").to_crs(epsg=3035)


# Create subplots: 1 row and 4 columns
fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=300)

# List of columns and titles
columns = ['NSE_single', 'NSE_glob_dynonly', 'NSE_glob_dynstat', 'NSE dynstat - single']
titles = ['NSE Single', 'NSE Global Dyn only', 'NSE Global Dyn + Stat', 'Delta NSE']

# Set colormap to use
cmap = 'RdBu'

# Loop through columns to create each plot
for ax, column, title in zip(axes, columns, titles):
    # Plot Bundeslaender boundaries
    bundeslaender2.plot(ax=ax, facecolor='lightgrey', edgecolor='darkgrey', alpha=0.5, linewidth=1)
    
    # Plot points with color mapping based on the current column
    scatter = gdf.plot(ax=ax, column=column, cmap=cmap, marker='o', 
                       markersize=3, alpha=1, legend=False, vmin=-1, vmax=1)
    
    # Set plot title
    ax.set_title(title, fontsize=12, y=1.05)
    
    # Turn off axis labels and ticks
    ax.set_axis_off()

# Add a shared colorbar for subplots 1 to 3
norm = plt.Normalize(vmin=-1, vmax=1)  # Normalize data between vmin and vmax
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes[:3], orientation='horizontal', fraction=0.02, extend='min')

# Move the shared colorbar under the second subplot (middle of 1-3)
cbar.ax.get_xaxis().labelpad = 10  # Add extra padding to the colorbar label
cbar.ax.set_position([0.27, 0.05, 0.3, 0.03])  # [x, y, width, height] in figure coordinates
cbar.set_label('NSE', fontsize=10)  # Set the label for the shared colorbar
cbar.set_ticks([-1, 0, 1])  # Set the tick positions based on the range

# Add a separate colorbar for the 4th subplot
cbar4 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=axes[3], orientation='vertical', fraction=0.032, extend='both')

# Adjust colorbar4's position
cbar4.ax.get_xaxis().labelpad = 10  # Add extra padding to the colorbar label
cbar4.ax.set_position([0.75, 0.05, 0.3, 0.03])  # [x, y, width, height] in figure coordinates
cbar4.set_label('Delta NSE', fontsize=10)  # Label for subplot 4's colorbar
cbar4.set_ticks([-1, 0, 1])  # Set the tick positions for subplot 4

# Adjust layout: reduce space between subplots and adjust colorbar position
plt.subplots_adjust(wspace=-0.2)  # Adjust space between subplots

# # Export to PDF
#plt.tight_layout()
plt.savefig("nse_maps.pdf", format='pdf')  # Save as PDF

# Show plot
plt.show()


