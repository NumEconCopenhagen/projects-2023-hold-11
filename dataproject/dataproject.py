import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
#%pip install matplotlib-venn
from matplotlib_venn import venn2
from statsmodels.tsa.statespace.sarimax import SARIMAX

filename = 'data/Bachelor-data.xlsx' # open the file (read)
df = pd.read_excel(filename)
df.rename(columns={'Unnamed: 0':'year'}, inplace = True) # rename the unnamed year column to year (clean)


# this code is used for skipping quarters in the plots to avoid to many entries on the x-axis
l =[]
for j, i in enumerate(df.year.values):

    if j%4==0 :
        pass
    else:
        i = '' 
    l.append(i)
l = np.array(l)    


# function that takes a dataframe and creates a plot
def _plot_timeseries(dataframe, variable):
    
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    variable = list(variable)
    
    x = dataframe.year.values
    y = dataframe[variable].values
    

    ax.plot(x,y, label = variable)
    if len(variable) != 1:

        title = ' and '.join(variable)
    else:
        title = variable[0]
    ax.set_title(title)
    ax.set_xticklabels(l,rotation= 90)
    ax.legend(loc='upper right')
    plt.show()  

# plots the plots interactivly     
def plot_timeseries(dataframe):
    """plot the time series with interactions"""
    
    widgets.interact(_plot_timeseries, 
    dataframe = widgets.fixed(dataframe),
    variable = widgets.SelectMultiple(
        description='variable', 
        options=['Unemployment rate','Total employment, growth','Central bank key interest rate','CPI','Private consumption, growth','Private final consumption, volume', 'Government consumption, growth', 'Government consumption, volume', 'GDP, growth','GDP, volume, market prices','taxes', 'CPI, growth'], 
        value=['CPI']),
    ); 

def phillips_curve(a = 2, slope= -2):
    """Plots the phillips curve"""
    # Define the Phillips curve parameters
    a = a
    slope = slope

    # Define the range of values for unemployment rate
    unemployment_rates = np.linspace(0, 14, 100)

    # Calculate the corresponding inflation rates using the Phillips curve equation
    inflation_rates = a + slope * np.log(unemployment_rates)

    # Create the plot
    plt.plot(unemployment_rates, inflation_rates, label='Theoretical Phillips Curve', color = 'red')
    plt.xlabel('Unemployment Rate')
    plt.ylabel('Inflation Rate')
    plt.title('Theoretical Phillips Curve')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# Function to combine Phillips curve and scatter plot
def graph_combine(dataframe = df):
    """Plots the Phillips curve and scatter plot"""
    fig, ax = plt.subplots()
    
    # Plot the scatter plot
    dataframe.plot.scatter(x='Unemployment rate', y='CPI, growth', ax=ax, title='Swedish Phillips-curve 1990Q1-2020Q1', label = 'actual data')
    
    # Plot the Phillips curve
    phillips_curve()    
    # Set labels and legend
    ax.set_xlabel('Unemployment Rate')
    ax.set_ylabel('Inflation Rate')
    ax.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()

