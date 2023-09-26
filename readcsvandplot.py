import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler




##########################################################################################################################################################################
#
#
#              Read Greedy CSV file
#
#
##########################################################################################################################################################################
filename1 = "CSVs/outputgreedy.csv"

# Read the CSV data into a list of lists
data1 = []
with open(filename1, "r") as file:
    for line in file:
        row = line.strip().split(',')
        # if len(row) > 55:  # Change 49 to the expected column count
        #     print(row) 
        data1.append(row)

# Find the maximum number of columns in any row
max_columns1 = max(len(row) for row in data1)

# Fill missing values in rows to match the maximum number of columns
for row in data1:
    row.extend([''] * (max_columns1 - len(row)))



header1 = [
    
    'N_V', 'N_R', 'N_T', 'Counter',
    'minC1_up', 'avgC1_up', 'maxC1_up',    #Residual or unoccupied data rates for Optical APs upload
    'minC2_up', 'avgC2_up', 'maxC2_up',    #Residual or unoccupied data rates for radio APs upload
    'unused_up_vlc', 'unused_up_rf',    #unused_ap  
    'minC1_down', 'avgC1_down', 'maxC1_down',    #Residual or unoccupied data rates for Optical APs downlinks
    'minC2_down', 'avgC2_down', 'maxC2_down',    #Residual or unoccupied data rates for radio APs downlinks
    'unused_down_vlc', 'unused_down_rf',    #unused_ap  
    
]

# Create a DataFrame from the data
newdf = pd.DataFrame(data1, columns=header1)
newdf = newdf.drop('Counter', axis=1)
newdf = newdf.apply(pd.to_numeric, errors='coerce')
newdf = newdf[newdf['N_T'] <= 25]
newdf.reset_index(drop=True, inplace=True)


##########################################################################################################################################################################
#
#
#              Read MA_TD CSV file
#
#
##########################################################################################################################################################################
filename = "CSVs/output_server.csv"

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
    'minC6_up', 'avgC6_up', 'maxC6_up',    #Distances between nodes and optical APs in uplink 
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
df = df[df['N_T'] <= 25]
df.reset_index(drop=True, inplace=True)
df.fillna(method='ffill', inplace=True)


##########################################################################################################################################################################
#
#
#              plot availbable data rate
#
#
##########################################################################################################################################################################

def data_rate(df, stat1):
    df['active_rf'] = np.maximum(df['N_R'] - df['unused_rf'], 0)
    df['active_vlc'] = np.maximum(df['N_V'] - df['unused_vlc'], 0)

    if 'C1' in stat1:
        Nn = 'N_V'
        Nunused = 'active_vlc'
        Ylable = 'Active Optical APs'
        Nn2 = '$N_V$'
    elif 'C2' in stat1:
        Nn = 'N_R'
        Nunused = 'active_rf'
        Ylable = 'Active Radio APs'
        Nn2 = '$N_R$'
    if 'up' in stat1:
        legen = ' of uplink data rates'
    else:
        legen = ' of downlink data rates'

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    
    
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    df[['min' + stat1, 'avg' +stat1, 'max' + stat1]] = scaler.fit_transform(df[['min' + stat1, 'avg' +stat1, 'max' + stat1]])

    result_dict = df.groupby(['N_T', Nn])['min' + stat1].mean().to_dict()
    # Print the resulting DataFrame with the averages
    x_values = [(N_T, N_V) for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    y_values_minC1_up = [value for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    # Create a line chart
    plt.plot(range(len(x_values)), y_values_minC1_up, marker='o', linestyle='-', color='g', label='Min' +legen)


    result_dict = df.groupby(['N_T', Nn])['avg' +stat1].mean().to_dict()
    # Print the resulting DataFrame with the averages
    y_values_avgC1_up = [value for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    # Create a line chart
    plt.plot(range(len(x_values)), y_values_avgC1_up, marker='s', linestyle='-', color='k', label='Mean' +legen)


    result_dict = df.groupby(['N_T', Nn])['max' + stat1].mean().to_dict()
    # Print the resulting DataFrame with the averages
    y_values_maxC1_up = [value for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    # Create a line chart
    plt.plot(range(len(x_values)), y_values_maxC1_up, marker='<', linestyle='-', color='r', label='Max' +legen)

    plt.legend()
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontweight='bold')
    variable_label = r'$\mathbf{(N_V' if Nn == 'N_V' else r'$\mathbf{(N_R'
    variable_label += r', N_T)}$ Pairs'
    plt.xlabel(variable_label, fontweight='bold')
    
    plt.ylabel('Average of available data rate', fontweight='bold')

    plt.ylim(0,100)

    # Create a second y-axis for 'unused_vlc' and 'unused_rf'
    ax2 = ax1.twinx()
    avg_unused = df.groupby(['N_T',Nn])[Nunused].mean().reset_index()
    
    median_Nunused = len(x_values)/2
    avg_unused['factor'] = avg_unused[Nunused].apply(lambda x: linear_interpolation_factor(median_Nunused, x, 1, 0.5))
    
    

    ax2.plot(range(len(x_values)), avg_unused['factor'] * avg_unused[Nunused], marker='D', linestyle='--', color='b', label=Nn2)
    ax2.yaxis.set_tick_params(labelcolor='blue')
    ax2.spines['right'].set_color('blue')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax2.set_ylabel(Ylable, color='b', fontweight='bold')
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, 15)
    
    
    y_axis = plt.gca().yaxis

    # Set the Y-axis tick labels to be bold
    for tick in y_axis.get_major_ticks():
        tick.label.set_fontweight('bold')

    # Set the x-axis labels and show the plot
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontweight='bold')
    plt.grid(True, axis='both')
    ax1.xaxis.grid(True, zorder=-1)
    plt.tight_layout()
    plt.savefig('Figs/minavgmax' + stat1 + '.png')
    #plt.show()

def linear_interpolation_factor(median_Nunused, Nunused, start_value=1, end_value=0.4):
    # Ensure that the factor is clamped between start_value and end_value
    
    return max(end_value, min(start_value - (start_value - end_value) / median_Nunused * Nunused, start_value))

def data_rate2(df, stat1):

    if 'C1' in stat1:
        Nn = 'N_V'
        Nunused = 'unused_vlc'
        Ylable = '#Active Optical APs'
        Nn2 = '$N_V$'
        if 'up' in stat1:
            Nunused = 'unused_up_vlc'
        else:
            Nunused = 'unused_down_vlc'
        
    elif 'C2' in stat1:
        Nn = 'N_R'
        Ylable = '#Active Radio APs'
        Nn2 = '$N_R$'
        if 'up' in stat1:
            Nunused = 'unused_up_rf'
        else:
            Nunused = 'unused_down_rf'

    if 'up' in stat1:
        legen = ' of uplink data rates'
    else:
        legen = ' of downlink data rates'

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    
    
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    df[['min' + stat1, 'avg' +stat1, 'max' + stat1]] = scaler.fit_transform(df[['min' + stat1, 'avg' +stat1, 'max' + stat1]])

    result_dict = df.groupby(['N_T', Nn])['min' + stat1].mean().to_dict()
    # Print the resulting DataFrame with the averages
    x_values = [(N_T, N_V) for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    y_values_minC1_up = [value for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    # Create a line chart
    plt.plot(range(len(x_values)), y_values_minC1_up, marker='o', linestyle='-', color='g', label='Min' +legen)


    result_dict = df.groupby(['N_T', Nn])['avg' +stat1].mean().to_dict()
    # Print the resulting DataFrame with the averages
    y_values_avgC1_up = [value for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    # Create a line chart
    plt.plot(range(len(x_values)), y_values_avgC1_up, marker='s', linestyle='-', color='k', label='Mean' +legen)


    result_dict = df.groupby(['N_T', Nn])['max' + stat1].mean().to_dict()
    # Print the resulting DataFrame with the averages
    y_values_maxC1_up = [value for (N_T, N_V), value in result_dict.items() if not np.isnan(value)]
    # Create a line chart
    plt.plot(range(len(x_values)), y_values_maxC1_up, marker='<', linestyle='-', color='r', label='Max' +legen)

    plt.legend()
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontweight='bold')
    variable_label = r'$\mathbf{(N_V' if Nn == 'N_V' else r'$\mathbf{(N_R}'
    variable_label += r', N_T)}$ Pairs'
    plt.xlabel(variable_label, fontweight='bold')
    plt.ylabel('Average of available data rate', fontweight='bold')

    plt.ylim(0,100)

    # Create a second y-axis for 'unused_vlc' and 'unused_rf'
    ax2 = ax1.twinx()
    avg_unused = df.groupby(['N_T',Nn])[Nunused].mean().reset_index()

    ax2.plot(range(len(x_values)), avg_unused[Nunused], marker='D', linestyle='--', color='b', label=Nn2)
    ax2.yaxis.set_tick_params(labelcolor='blue')
    ax2.spines['right'].set_color('blue')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax2.set_ylabel(Ylable, color='b', fontweight='bold')
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, 15)

    # Set the x-axis labels and show the plot
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontweight='bold')
    
    y_axis = plt.gca().yaxis

    # Set the Y-axis tick labels to be bold
    for tick in y_axis.get_major_ticks():
        tick.label.set_fontweight('bold')
        
    plt.grid(True, axis='both')
    ax1.xaxis.grid(True, zorder=-1)
    plt.tight_layout()
    plt.savefig('Figs/ggminavgmax' + stat1 + '.png')
    #plt.show()




stats = ['C1_up', 'C1_down', 'C2_up', 'C2_down']
for stat1 in stats:
    data_rate(df, stat1)
    data_rate2(newdf, stat1)




##########################################################################################################################################################################
#
#
#              Print unique counts of upcount and downcount
#
#
##########################################################################################################################################################################
unique_N_V = df['N_V'].unique()
unique_N_R = df['N_R'].unique()
unique_N_T = df['N_T'].unique()


upcount_rows_with_nan = len(df[df['UpCount'].isna()])
downcount_rows_with_nan = len(df[df['DownCount'].isna()])

print(f'unique_N_V: {unique_N_V}, unique_N_R: {unique_N_R}, unique_N_T: {unique_N_T}, upcount_rows_with_nan: {upcount_rows_with_nan}, downcount_rows_with_nan:{downcount_rows_with_nan}')




##########################################################################################################################################################################
#
#
#              plot uplink and downlink possibilities
#
#
##########################################################################################################################################################################
def plot_count_linechart_dif(df, dir):
    df = df[((df['N_V'] < 8) & (df['N_R'] < 8)) ]
    # Create subplots with three columns (for three figures side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Set common labels and title
    fig.suptitle('Grouped Line Chart of UpCount by $N_T$, $N_R$, and $N_V$')
    fig.text(0.5, 0.04, '$N_T$', ha='center')
    fig.text(0.04, 0.5, dir, va='center', rotation='vertical')

    # Loop through the different aggregation functions and create plots
    aggregation_functions = ['min', 'mean', 'max']
    for i, aggfunc in enumerate(aggregation_functions):
        pivot_df = df.pivot_table(index='N_T', columns=['N_R', 'N_V'], values=dir, aggfunc=aggfunc)
        ax = axes[i]
        for column in pivot_df.columns:
            ax.plot(unique_N_T, pivot_df[column].loc[unique_N_T], 'o-', linewidth=2.0, markersize=10, label=column)
        ax.set_title(f'Aggregation: {aggfunc}')
        ax.legend(title=('$N_R$', '$N_V$'))
        ax.set_xticks(unique_N_T)
        ax.set_xticklabels(unique_N_T, rotation=0)
        ax.set_yscale('log')

    # Save and display the figure
    plt.subplots_adjust(wspace=0.3)  # Adjust the horizontal space between subplots
    plt.savefig(f'Figs/{dir}.png')
    #plt.show()


def plot_count_linechart(df, dir):
    df = df[((df['N_V'] < 8) & (df['N_R'] < 8)) ]
    # Pivot the data using pivot_table and aggregate with mean
    pivot_df = df.pivot_table(index='N_T', columns=['N_R', 'N_V'], values=dir, aggfunc='max')

    # Define the x-values (N_T values)
    unique_N_T = df['N_T'].unique()

    # Plot the grouped line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.xlabel('$N_T$')
    plt.ylabel(dir)

    for column in pivot_df.columns:
        ax.plot(unique_N_T, pivot_df[column].loc[unique_N_T], 'o-', linewidth=2.0, markersize=10, label=column)

    for column in pivot_df.columns:
        values = pivot_df[column].loc[unique_N_T].tolist()  # Convert the values to a list
        values_str = ', '.join(map(str, values))  # Convert values to strings and join with commas
        print(f'{column}: {values_str}')

    plt.legend(title=('$N_R$', '$N_V$'))
    plt.xticks(unique_N_T, rotation=0)
    plt.yscale('log')
    plt.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('Figs/' + dir + '.png')
    #plt.show()

#Un comment the below lines to have the count of upcount and downcount topologies
#plot_count_linechart(df1, 'UpCount')
#plot_count_linechart(df1, 'DownCount')



##########################################################################################################################################################################
#
#
#              plot count of NaN in each states
#
#
##########################################################################################################################################################################
    
def count_NaN_linechart(df, dir):
    df = df[((df['N_V'] < 8) & (df['N_R'] < 8)) ]
    # Group the data by N_T, N_R, and N_V and count NaN values within each group
    df.sort_values(by='N_T', inplace=True)
    grouped_df = df.groupby(['N_T', 'N_R', 'N_V'])[dir].apply(lambda x: x.isna().sum()).reset_index()
    
    # Pivot the grouped data to create a grouped bar chart
    pivot_df = grouped_df.pivot(index='N_T', columns=['N_R', 'N_V'], values=dir)
    
    # Reindex the pivot_df to ensure all N_T values are present
    unique_N_T = df['N_T'].unique()
    pivot_df = pivot_df.reindex(unique_N_T)
    
    

    # Plot the grouped line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.xlabel('$N_T$')
    plt.ylabel(dir)

    for column in pivot_df.columns:
        ax.plot(unique_N_T, pivot_df[column].loc[unique_N_T], 'o-', linewidth=2.0, markersize=10, label=column)

    plt.legend(title=('$N_R$', '$N_V$'))
    plt.xticks(unique_N_T, rotation=0)
    plt.yscale('linear')
    plt.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('Figs/nan' + dir + '.png')
    #plt.show()
    

#Uncomment these line to have  the count of NaN in each states
#count_NaN_linechart(df1, 'UpCount')
#count_NaN_linechart(df1, 'DownCount')




##########################################################################################################################################################################
#
#
#              Save to CSV : deletenan.csv and average.csv
#
#
##########################################################################################################################################################################
def save_avg_csv(df1):
    df = df1.dropna()
    df.to_csv('CSVs/deletednan.csv')
    df = df.reset_index(drop=True)
    # Pivot the DataFrame
    pivot_df = df.pivot_table(index=['N_V', 'N_R', 'N_T'], aggfunc='mean')

    # Save the pivoted DataFrame to a CSV file
    pivot_df.to_csv('CSVs/averages.csv')


save_avg_csv(df)



##########################################################################################################################################################################
#
#
#              plot box chart for availbable data rate
#
#
##########################################################################################################################################################################
def plot_margins(df):
    df['usedap_rf'] = df['N_R'] - df['unused_rf']
    df['usedap_vlc'] = df['N_V'] - df['unused_vlc']
    df  = df[((df['usedap_vlc'] > 0) & (df['usedap_rf'] > 0)) ]

    scaler = MinMaxScaler(feature_range=(0, 100))
    df[['avgC2_up', 'avgC1_up', 'avgC2_down', 'avgC1_down']] = scaler.fit_transform(df[['avgC2_up', 'avgC1_up', 'avgC2_down', 'avgC1_down']])
    
    #stat1 = ['min', 'avg', 'max']
    stat1 = ['avg']
    for stat in stat1:
        
        
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C2_up', hue='usedap_rf', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active radio AP')
        plt.legend(title='$N_R$', title_fontsize='12')
        plt.grid(True)
        plt.savefig('Figs/' + stat + 'C2_up_vs_used_rf' + '.png')
        plt.show()
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C1_up', hue='usedap_vlc', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active optical AP')
        plt.legend(title='$N_V$', title_fontsize='12')
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C1_up_vs_used_vlc' + '.png')
        plt.show()
        
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C2_down', hue='usedap_rf', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active radio AP')
        plt.legend(title='$N_R$', title_fontsize='12')
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C2_down_vs_used_rf' + '.png')
        plt.show()
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C1_down', hue='usedap_vlc', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active optical AP')
        plt.legend(title='$N_V$', title_fontsize='12')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.grid(True)
        plt.savefig('Figs/' + stat + 'C1_down_vs_used_vlc' + '.png')
        plt.show()


#plot_margins(df1)


def plot_margins2(df):

    # Melt the DataFrame to reshape it for the boxplot
    melted_df_C2_up = pd.melt(df, id_vars=['N_V', 'N_R', 'N_T'], value_vars=['minC2_up', 'avgC2_up', 'maxC2_up'], var_name='Data_Type', value_name='Data_Value')

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_df_C2_up, x='N_R', y='Data_Value', hue='N_T')
    plt.xlabel('$N_R$')
    plt.ylabel('Data rate margin (Mbps) per radio AP')
    plt.legend(title='$N_T$', title_fontsize='12')
    plt.grid(True)
    plt.ylim(0, 110)
    plt.savefig('Figs/' +'C2_up.png')
    
    melted_df_C2_down = pd.melt(df, id_vars=['N_V', 'N_R', 'N_T'], value_vars=['minC2_down', 'avgC2_down', 'maxC2_down'], var_name='Data_Type', value_name='Data_Value')

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_df_C2_down, x='N_R', y='Data_Value', hue='N_T')
    plt.xlabel('$N_R$')
    plt.ylabel('Data rate margin (Mbps) per radio AP')
    plt.legend(title='$N_T$', title_fontsize='12')
    plt.grid(True)
    plt.ylim(0, 550)
    plt.savefig('Figs/' +'C2_down.png')


    # Melt the DataFrame to reshape it for the boxplot
    melted_df_C1_up = pd.melt(df, id_vars=['N_V', 'N_R', 'N_T'], value_vars=['minC1_up', 'avgC1_up', 'maxC1_up'], var_name='Data_Type', value_name='Data_Value')

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_df_C1_up, x='N_V', y='Data_Value', hue='N_T')
    plt.xlabel('$N_V$')
    plt.ylabel('Data rate margin (Mbps) per optical AP')
    plt.legend(title='$N_T$', title_fontsize='12')
    plt.grid(True)
    plt.ylim(0, 550)
    plt.savefig('Figs/' +'C1_up.png')
    
    melted_df_C1_down = pd.melt(df, id_vars=['N_V', 'N_R', 'N_T'], value_vars=['minC1_down', 'avgC1_down', 'maxC1_down'], var_name='Data_Type', value_name='Data_Value')

    # Create a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted_df_C1_down, x='N_V', y='Data_Value', hue='N_T')
    plt.xlabel('$N_V$')
    plt.ylabel('Data rate margin (Mbps) per optical AP')
    plt.legend(title='$N_T$', title_fontsize='12')
    plt.grid(True)
    plt.ylim(0, 1100)
    plt.savefig('Figs/' +'C1_down.png')

#plot_margins2(df)

def plot_marginsn(df):
    df['usedap_rf'] = df['N_R'] - df['unused_rf']
    df['usedap_vlc'] = df['N_V'] - df['unused_vlc']
    df  = df[((df['usedap_vlc'] > 0) & (df['usedap_rf'] > 0)) ]

    scaler = MinMaxScaler(feature_range=(0, 100))
    df[['avgC2_up', 'avgC1_up', 'avgC2_down', 'avgC1_down']] = scaler.fit_transform(df[['avgC2_up', 'avgC1_up', 'avgC2_down', 'avgC1_down']])
    
    #stat1 = ['min', 'avg', 'max']
    stat1 = ['avg']
    for stat in stat1:
        
        

        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C2_up', hue='usedap_rf', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active radio AP')
        plt.legend(title='$N_R$', title_fontsize='12')
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C2_up_vs_used_rfgg' + '.png')
        plt.show()
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C1_up', hue='usedap_vlc', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active optical AP')
        plt.legend(title='$N_V$', title_fontsize='12')
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C1_up_vs_used_vlcgg' + '.png')
        plt.show()
        
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C2_down', hue='usedap_rf', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active radio AP')
        plt.legend(title='$N_R$', title_fontsize='12')
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C2_down_vs_used_rfgg' + '.png')
        plt.show()
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_T', y=stat + 'C1_down', hue='usedap_vlc', showfliers=False)
        plt.xlabel('$N_T$')
        plt.ylabel('Available Data rate (Mbps) of active optical AP')
        plt.legend(title='$N_V$', title_fontsize='12')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C1_down_vs_used_vlcgg' + '.png')
        plt.show()

#plot_marginsn(df1)



##########################################################################################################################################################################
#
#
#              plot box chart for distances
#
#
##########################################################################################################################################################################
def plot_distances():
    
    stat1 = ['min', 'avg', 'max']
    for stat in stat1:
        
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_R', y=stat + 'C7_up', hue='N_T')
        plt.xlabel('$N_R$')
        plt.ylabel(stat + 'C7_up')
        plt.legend(title='$N_T$', title_fontsize='12')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C7_up' + '.png')
        #plt.show()
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_V', y=stat + 'C6_up', hue='N_T')
        plt.xlabel('$N_V$')
        plt.ylabel(stat + 'C6_up')
        plt.legend(title='$N_T$', title_fontsize='12')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C6_up' + '.png')
        #plt.show()
        
        # Create a boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_R', y=stat + 'C7_down', hue='N_T')
        plt.xlabel('$N_R$')
        plt.ylabel(stat + 'C7_down')
        plt.legend(title='$N_T$', title_fontsize='12')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C7_down' + '.png')
        #plt.show()
        
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='N_V', y=stat + 'C6_down', hue='N_T')
        plt.xlabel('$N_V$')
        plt.ylabel(stat + 'C6_down')
        plt.legend(title='$N_T$', title_fontsize='12')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.grid(True)
        plt.savefig('Figs/' +stat + 'C6_down' + '.png')
        #plt.show()
#plot_distances()





##########################################################################################################################################################################
#
#
#              plot heat chart for distances
#
#
##########################################################################################################################################################################

def plot_heat_distances():

    # Pivot the DataFrame to create a matrix of 'N_R' vs. 'N_T' values
    pivot_data = df.pivot_table(index='N_R', columns='N_T', values='minC7_down', aggfunc='median')
    
    # Create a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)

    # Customize the plot
    plt.xlabel('$N_T$')
    plt.ylabel('$N_R$')
    plt.savefig('Figs/' +'C7_down.png')
    # Show the plot
    plt.show()

    # Pivot the DataFrame to create a matrix of 'N_R' vs. 'N_T' values
    pivot_data = df.pivot_table(index='N_R', columns='N_T', values='minC7_up', aggfunc='median')
    
    # Create a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)

    # Customize the plot
    plt.xlabel('$N_T$')
    plt.ylabel('$N_R$')
    plt.savefig('Figs/' +'C7_up.png')
    # Show the plot
    plt.show()


    pivot_data = df.pivot_table(index='N_V', columns='N_T', values='minC6_down', aggfunc='median')
    
    # Create a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)

    # Customize the plot
    plt.xlabel('$N_T$')
    plt.ylabel('$N_V$')
    plt.savefig('Figs/' +'C6_down.png')
    # Show the plot
    plt.show()

    # Pivot the DataFrame to create a matrix of 'N_R' vs. 'N_T' values
    pivot_data = df.pivot_table(index='N_V', columns='N_T', values='minC6_up', aggfunc='median')
    
    # Create a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)

    # Customize the plot
    plt.xlabel('$N_T$')
    plt.ylabel('$N_V$')
    plt.savefig('Figs/' +'C6_up.png')
    # Show the plot
    plt.show()
#plot_heat_distances()



def dis_line():
    #plot_heat_distances()
    plt.figure(figsize=(12, 6))
    df['avgC6_down_per_unused_vlc'] = df['avgC6_down'] / (df['N_V'] - df['unused_vlc'])

    df_grouped = df.groupby('N_T')['avgC6_down_per_unused_vlc'].mean()

    plt.plot(df_grouped.index, df_grouped.values, 'o-', linewidth=2.0, markersize=10, label='C6_down_per_used_vlc')
    
    

    df['avgC6_up_per_unused_vlc'] = df['avgC6_up'] / (df['N_V'] - df['unused_vlc'])

    df_grouped = df.groupby('N_T')['avgC6_up_per_unused_vlc'].mean()

    plt.plot(df_grouped.index, df_grouped.values, 'o-', linewidth=2.0, markersize=10, label='C6_up_per_used_vlc')
    
    


    df['avgC7_down_per_unused_rf'] = df['avgC7_down'] / (df['N_R'] - df['unused_rf'])

    df_grouped = df.groupby('N_T')['avgC7_down_per_unused_rf'].mean()

    plt.plot(df_grouped.index, df_grouped.values, 'o-', linewidth=2.0, markersize=10, label='C7_down_per_used_rf')
    
    

    df['avgC7_up_per_unused_rf'] = df['avgC7_up'] / (df['N_R'] - df['unused_rf'])

    df_grouped = df.groupby('N_T')['avgC7_up_per_unused_rf'].mean()

    plt.plot(df_grouped.index, df_grouped.values, 'o-', linewidth=2.0, markersize=10, label='C7_up_per_used_rf')
    plt.xlabel('$N_T$')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.savefig('Figs/' +'dist_vs_Num_APs.png')
    #plt.show()

#dis_line()



##########################################################################################################################################################################
#
#
#              plot active APs
#
#
##########################################################################################################################################################################
def unused():
    # Calculate 'used_vlc' and 'used_rf'
    df['used_vlc'] = df['N_V'] - df['unused_vlc']
    df['used_rf'] = df['N_R'] - df['unused_rf']

    # Group by 'N_T' and calculate the median for 'used_vlc' and 'used_rf'
    median_data = df.groupby('N_T')[['used_vlc', 'used_rf']].mean().reset_index()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(median_data['N_T'], median_data['used_vlc'], label='$N_V$', marker='o')
    plt.plot(median_data['N_T'], median_data['used_rf'], label='$N_R$', marker='o')
    plt.xlabel('$N_T$')
    plt.ylabel('# APs')
    tick = [3,5,10,15,20]
    plt.xticks(tick, rotation=0)
    plt.legend()
    plt.grid(True)
    plt.savefig('Figs/' +'usedaps.png')
    plt.show()

#unused()