import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plot 
import sort_data as sd

PKEYS = ['threshold','death_rate']

raw_data = sd.load_data('CIP_data_fixed_cc_rates_fixed_size')
axs = plot.plot_equilibriums_same_axis(raw_data,'pop_size',200.,show_error=False,ylabel="Population size")
axs = plot.plot_equilibriums_same_axis(raw_data,'density',200.,show_error=False,ylabel="Density")
axs = plot.plot_equilibriums_same_axis(raw_data,'cell_seperation',200.,show_error=False,ylabel="Cell seperation")