import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
import numpy as np
import seaborn as sns

# Sample CSV data
filename = "output.csv"

# Read the CSV data into a list of lists
data = []
with open(filename, "r") as file:
    for line in file:
        row = line.strip().split(',')
        # if len(row) > 55:  # Change 49 to the expected column count
        #     print(row) 
        data.append(row)

# Find the maximum number of columns in any row
max_columns = max(len(row) for row in data)

# Fill missing values in rows to match the maximum number of columns
for row in data:
    row.extend([''] * (max_columns - len(row)))



header = [
    
    'N_V', 'N_R', 'N_T', 'Counter',
    'minC1_up', 'avgC1_up', 'maxC1_up',    #Residual or unoccupied data rates for Optical APs upload
    'minC2_up', 'avgC2_up', 'maxC2_up',    #Residual or unoccupied data rates for radio APs upload
    'minC4_up', 'avgC4_up', 'maxC4_up',    #unused_ap  
    'minC5_up', 'avgC5_up', 'maxC5_up',    #price: very high or low for uplinks  rate of VLC to radio
    'minC6_up', 'avgC6_up', 'maxC6_up',    #Distances between nodes and radio APs in uplink 
    'minC7_up', 'avgC7_up', 'maxC7_up',    #Distances between nodes and radio APs in uplink 
    'minC1_down', 'avgC1_down', 'maxC1_down',    #Residual or unoccupied data rates for Optical APs downlinks
    'minC2_down', 'avgC2_down', 'maxC2_down',    #Residual or unoccupied data rates for radio APs downlinks
    'minC4_down', 'avgC4_down', 'maxC4_down',    #unused_ap  
    'minC5_down', 'avgC5_down', 'maxC5_down',    #price: very high or low for downlinks
    'minC6_down', 'avgC6_down', 'maxC6_down',    #Distances between nodes and optical APs in downlinks 
    'minC7_down', 'avgC7_down', 'maxC7_down',    #Distances between nodes and radio APs in downlinks 
    'UpCount' , 'DownCount', 'Total_Count', 'ex_time',
    'unused_vlc', 'unused_rf',
]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=header)
df = df.drop('Counter', axis=1)
df = df.apply(pd.to_numeric, errors='coerce')

# # Print the DataFrame
#print(tabulate(df, headers='keys', tablefmt='fancy_grid'))



unique_N_V = df['N_V'].unique()
unique_N_R = df['N_R'].unique()
unique_N_T = df['N_T'].unique()


upcount_rows_with_nan = len(df[df['UpCount'].isna()])
downcount_rows_with_nan = len(df[df['DownCount'].isna()])

print(f'unique_N_V: {unique_N_V}, unique_N_R: {unique_N_R}, unique_N_T: {unique_N_T}, upcount_rows_with_nan: {upcount_rows_with_nan}, downcount_rows_with_nan:{downcount_rows_with_nan}')


def plot_count_barchart_up():
    # Pivot the data using pivot_table and aggregate with sum
    pivot_df = df.pivot_table(index='N_T', columns=['N_R', 'N_V'], values='UpCount', aggfunc='mean')

    # Plot the grouped bar chart
    ax = pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('$N_T$')
    plt.ylabel('UpCount')
    ax.set_yscale('log')
    plt.title('Grouped Bar Chart of UpCount by $N_T$, $N_R$, and $N_V$')

    plt.legend(title=('$N_R$', '$N_V$'))
    plt.xticks(np.arange(len(pivot_df.index)), pivot_df.index, rotation=0)
    plt.savefig('upcount.png')
    plt.show()
    

def plot_count_barchart_down():
    # Pivot the data using pivot_table and aggregate with sum
    pivot_df = df.pivot_table(index='N_T', columns=['N_R', 'N_V'], values='DownCount', aggfunc='mean')

    # Plot the grouped bar chart
    ax = pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('$N_T$')
    plt.ylabel('DownCount')
    ax.set_yscale('log')
    plt.title('Grouped Bar Chart of downCount by $N_T$, $N_R$, and $N_V$')

    plt.legend(title=('$N_R$', '$N_V$'))
    plt.xticks(np.arange(len(pivot_df.index)), pivot_df.index, rotation=0)
    plt.savefig('downcount.png')
    plt.show()
    
    
def count_NaN_upcount():
    # Group the data by N_T, N_R, and N_V and count NaN values within each group
    grouped_df = df.groupby(['N_T', 'N_R', 'N_V'])['UpCount'].apply(lambda x: x.isna().sum()).reset_index()

    # Pivot the grouped data to create a grouped bar chart
    pivot_df = grouped_df.pivot(index='N_T', columns=['N_R', 'N_V'], values='UpCount')

    # Plot the grouped bar chart
    ax = pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('$N_T$')
    plt.ylabel('Count of NaN UpCount')
    plt.title('Grouped Bar Chart of Count of NaN UpCount by $N_T$, $N_R$, and $N_V$')

    plt.legend(title=('$N_R$', '$N_V$'))
    plt.xticks(np.arange(len(pivot_df.index)), pivot_df.index, rotation=0)
    plt.savefig('nanupcount.png')
    plt.show()
    
    
def count_NaN_downcount():
    # Group the data by N_T, N_R, and N_V and count NaN values within each group
    grouped_df = df.groupby(['N_T', 'N_R', 'N_V'])['DownCount'].apply(lambda x: x.isna().sum()).reset_index()

    # Pivot the grouped data to create a grouped bar chart
    pivot_df = grouped_df.pivot(index='N_T', columns=['N_R', 'N_V'], values='DownCount')

    # Plot the grouped bar chart
    ax = pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('$N_T$')
    plt.ylabel('Count of NaN downCount')
    plt.title('Grouped Bar Chart of Count of NaN downCount by $N_T$, $N_R$, and $N_V$')

    plt.legend(title=('$N_R$', '$N_V$'))
    plt.xticks(np.arange(len(pivot_df.index)), pivot_df.index, rotation=0)
    plt.savefig('nandowncount.png')
    plt.show()
    

#Uncomment these line to have their plot
# plot_count_barchart_up()
# plot_count_barchart_down()
# count_NaN_upcount()
# count_NaN_downcount()


def save_avg_csv(df1):
    # Pivot the DataFrame
    pivot_df = df1.pivot_table(index=['N_V', 'N_R', 'N_T'], aggfunc='mean')

    # Save the pivoted DataFrame to a CSV file
    pivot_df.to_csv('averages.csv')

    # Display the pivoted DataFrame (optional)
    print(pivot_df)


df = df.dropna(subset=['UpCount', 'DownCount'])

# Reset the index of the DataFrame
df = df.reset_index(drop=True)
df.to_csv('deletednan.csv')
save_avg_csv(df)


usedap_rf = df['N_R'] - df['unused_rf']
df['minC2_up/usedap_vlc'] = df['minC2_up'] / usedap_rf

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='N_V', y='minC2_up/usedap_vlc', hue='N_T')
plt.xlabel('N_V')
plt.ylabel('minC2_up / unused_rf')
plt.title('Boxplot of minC2_up/unused_rf vs. N_V/N_T')
plt.legend(title='N_T', title_fontsize='12')
plt.grid(True)
plt.show()



df['avgC2_up/unused_rf'] = df['avgC2_up'] / usedap_rf

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='N_V', y='avgC2_up/unused_rf', hue='N_T')
plt.xlabel('N_V')
plt.ylabel('avgC1_up / usedap_vlc')
plt.title('Boxplot of avgC1_up/usedap_vlc vs. N_V/N_T')
plt.legend(title='N_T', title_fontsize='12')
plt.grid(True)
plt.show()



# Create separate figures for N_V, N_R, and N_T
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

# Plot minC6_up vs N_V
ax1.scatter(df['N_V'], df['minC6_up'], label='minC6_up vs N_V')
ax1.set_xlabel('N_V')
ax1.set_ylabel('minC6_up')
ax1.set_title('minC6_up vs N_V')

# Plot minC6_up vs N_R
ax2.scatter(df['N_R'], df['minC6_up'], label='minC6_up vs N_R')
ax2.set_xlabel('N_R')
ax2.set_ylabel('minC6_up')
ax2.set_title('minC6_up vs N_R')

# Plot minC6_up vs N_T
ax3.scatter(df['N_T'], df['minC6_up'], label='minC6_up vs N_T')
ax3.set_xlabel('N_T')
ax3.set_ylabel('minC6_up')
ax3.set_title('minC6_up vs N_T')

# Show the plots
plt.show()
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Extract the columns for the X, Y, and Z axes
# X = df['N_R']
# Y = df['N_T']
# Z1 = df['UpCount']

# # Create the 3D scatter plot
# ax.scatter(X, Y, Z1, c='r', marker='o')


# # Set labels for the axes
# ax.set_xlabel('N_R')
# ax.set_ylabel('N_T')
# ax.set_zlabel('UpCount')
# ax.set_zscale('linear')
# # Show the plot
# plt.show()


max_ex_time = df['ex_time'].max()
print("Maximum ex_time:", max_ex_time)

# Create a 3D plot
# Define the grid for the surface plot
# X = df['N_R']
# Y = df['N_T']

# # Interpolate the Z values based on your data
# Z = griddata((df['N_R'], df['N_T']), df['UpCount'], (X, Y), method='linear')

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create the 3D surface plot
# surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# # Set labels for the axes
# ax.set_xlabel('N_R')
# ax.set_ylabel('N_T')
# ax.set_zlabel('UpCount')

# # Add a color bar
# fig.colorbar(surf)

# # Show the plot
# plt.show()

# # # Convert 'N_V' to numeric (assuming it represents integers)
# # count_data['N_V'] = pd.to_numeric(count_data['N_V'])

# # # Create a bar plot
# # plt.figure(figsize=(10, 6))
# # plt.bar(count_data['N_V'], count_data['Counted'], color='skyblue')
# # plt.xlabel('$N_V$', fontsize=16)
# # plt.ylabel('Count of Data Points', fontsize=16)
# # plt.title('Count of Data Points where Count is not "N" vs. N_V', fontsize=16)
# # plt.xticks(count_data['N_V'], rotation=45)
# # plt.tight_layout()
# # plt.show()

# # #remove all rows with N values for count (no proper topologies)
# # df = df[df['Count'] != 'N']
# # #print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
# # df = df.apply(pd.to_numeric, errors='coerce')

# # column_names = df.columns

# # # Iterate through each column and convert to numeric
# # for col in column_names:
# #     df[col] = pd.to_numeric(df[col], errors='coerce')

# # N_v = df['N_V'].values
# # N_r = df['N_R'].values
# # N_t = df['N_T'].values
# # minC1 = df['minC1'].values
# # avgC1 = df['avgC1'].values
# # maxC1 = df['maxC1'].values

# # fig, ax = plt.subplots()  # Initialize the figure and axes
# # ax.plot(N_v + N_r + N_t, minC1, 'ro-', linewidth=2.0, markersize=10, label='min')
# # ax.plot(N_v + N_r + N_t, avgC1,  'b^-', linewidth=2.0, markersize=10, label='max')
# # ax.plot(N_v + N_r + N_t, maxC1,  'gs-', linewidth=2.0, markersize=10, label='avg')
# # #ax.plot(x, self.data[data_type][attr]['median'],  'kP-', linewidth=2.0, markersize=10, label='median')

# # ax.grid(True, which="both")
# # ax.set_xlabel('Iteration', fontsize=16)
# # ax.set_ylabel('Values')
# # ax.set_title('Values over Iterations')
# # ax.legend(prop={'size': 14})
# # plt.tight_layout()
# # #plt.savefig('newfig/2.png')
# # plt.show()





# # xv, yv = np.meshgrid(np.arange(df.shape[0]), np.arange(df.shape[1]))
# # column_names = df.columns.values

# # ma = np.nanmax(df.values)
# # norm = matplotlib.colors.Normalize(vmin=0, vmax=ma, clip=True)

# # fig = plt.figure(1)
# # ax = fig.add_subplot(111, projection='3d')  # Use add_subplot to specify 3D projection
# # surf = ax.plot_surface(xv, yv, df.values, cmap='viridis_r', linewidth=0.3, alpha=0.8, edgecolor='k', norm=norm)

# # plt.show()