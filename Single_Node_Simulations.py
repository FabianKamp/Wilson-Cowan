import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np 
from neurolib.models.wc import WCModel
import seaborn as sns
import itertools
from multiprocessing import Pool
from scipy.fft import fft

def evaluate_wc(params):
    """
    Function to run wc model with parameter settings
    """
    # Set seed
    np.random.seed(0)
    # Initialize the Wilson Cohan model
    wc = WCModel() 
    # Set the duration parameter two second
    timepoints = 2.*1000
    fsample = 1000.
    wc.params['duration'] = timepoints
    # Run model with parameter setting
    for key, value in params.items():
        wc.params[key] = value
    wc.run()
    exc_tc = wc.outputs.exc[0,100:]
    inh_tc = wc.outputs.inh[0,100:]
    # Get Outpus stats
    # Get min values
    max_val = np.max(exc_tc)
    min_val = np.min(exc_tc)
    # Get avg value
    avg_value = np.mean(exc_tc)
    # Get max freq greater than 0
    exc_f = fft(exc_tc)
    exc_f = 2.0/timepoints * np.abs(exc_f[0:int(timepoints/2)])
    freqs = np.linspace(0.0, (1.0/2.0)*fsample, int(timepoints/2))
    freqs = freqs[exc_f>1e-1]
    max_freq = freqs[-1]
    return (max_val, min_val, avg_value, max_freq)

def plot_4_params():
    """
    Function to plot the effect of 
    """
    # Noise Parameters
    nr_noise_levels = 1
    noise_levels = np.linspace(0.00,0.01,nr_noise_levels)
    # NMDA Parameters
    nr_parameter_levels = 2
    exc_inputs = np.linspace(0.5, 1, nr_parameter_levels)
    ei_couplings = np.linspace(9, 18, nr_parameter_levels)[::-1]
    # Gaba Parameters
    ii_couplings = np.linspace(0, 5, nr_parameter_levels)[::-1]
    ie_couplings = np.linspace(9, 18, nr_parameter_levels)
    pdf = PdfPages('testing2.pdf')
    # Noise loop
    for noise_level in noise_levels:
        # Set up figures
        diff_fig, diff_axes = plt.subplots(nr_parameter_levels, nr_parameter_levels, figsize=(15,15))
        diff_fig.suptitle(f'Max-Min Difference, Noise Level: {noise_level}', fontsize=25)
        fl_diff_ax = diff_axes.ravel()
        diff_cbar = diff_fig.add_axes([.92, 0.15, .03, 0.7])

        avg_fig, avg_axes = plt.subplots(nr_parameter_levels, nr_parameter_levels, figsize=(15,15))
        avg_fig.suptitle(f'Maximal Frequency, Noise Level: {noise_level}', fontsize=25)
        fl_avg_ax = avg_axes.ravel()
        avg_cbar = avg_fig.add_axes([.92, 0.15, .03, 0.7])
        
        freq_fig, freq_axes = plt.subplots(nr_parameter_levels, nr_parameter_levels, figsize=(15,15))
        freq_fig.suptitle(f'Maximal Frequency, Noise Level: {noise_level}', fontsize=25)
        fl_freq_ax = freq_axes.ravel()
        freq_cbar = freq_fig.add_axes([.92, 0.15, .03, 0.7])

        # Loop over NMDA parameters
        for m, (ei_c, e_input) in enumerate(itertools.product(ei_couplings, exc_inputs)):
            max_mat, min_mat, avg_mat, freq_mat=[np.zeros((nr_parameter_levels,nr_parameter_levels))]*4
            
            # Sets up the parameter for simulation
            params = [{'exc_ext':[e_input], 'c_excinh':ei_c, 'c_inhexc':ie_c, 'c_inhinh':ii_c, 'sigma_ou':noise_level} 
            for ie_c, ii_c in itertools.product(ie_couplings, ii_couplings)]           
            
            # Runs the model in parallel over Gaba parameter 
            with Pool(5) as p:
                result = p.map(evaluate_wc, params)
            max_values, min_values, avg_values, max_freq = zip(*result)                    
            max_mat.ravel()[:] = max_values
            min_mat.ravel()[:] = min_values
            avg_mat.ravel()[:] = avg_values
            freq_mat.ravel()[:] = max_freq
            diff_mat = max_mat-min_mat
            
            # Plot heatmap of diff and freq mat
            sns.heatmap(diff_mat, ax=fl_diff_ax[m], vmin=0, vmax=0.5, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else diff_cbar)
            sns.heatmap(avg_mat, ax=fl_avg_ax[m], vmin=0, vmax=0.5, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else diff_cbar)
            sns.heatmap(freq_mat, ax=fl_freq_ax[m], center=40, vmin=1, vmax=80, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else freq_cbar)
            
            # Configure axes
            for ax in [fl_diff_ax, fl_freq_ax, fl_avg_ax]:            
                ax[m].set_ylabel(f'E-I Coupl.: {np.round(ei_c, 2)}', labelpad=20, fontsize=15)
                ax[m].set_xlabel(f'Baseline Excitation: {np.round(e_input,2)}', labelpad=20, fontsize=15)
                ax[m].set_xticks(np.arange(0,nr_parameter_levels)+0.5)
                ax[m].set_yticks(np.arange(0,nr_parameter_levels)+0.5)
                ax[m].set_xticklabels([str(np.round(ie_c,1)) for ie_c in ie_couplings])
                ax[m].set_yticklabels([str(np.round(ii_c,1)) for ii_c in ii_couplings])
        
        # Configure colorbar 
        for cbar in [diff_cbar, freq_cbar, avg_cbar]:
            cbar.tick_params(labelsize=10, left=False, labelleft=False, right=True, labelright=True)
            #cbar.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for axes in [fl_diff_ax, fl_freq_ax, fl_avg_ax]:
            for ax in axes:
                ax.label_outer()

        # Save to pdf page
        pdf.savefig(diff_fig)
        pdf.savefig(avg_fig)
        pdf.savefig(freq_fig)
    
    pdf.close()

def plot_2_params(filename, **kwargs):
    """
    Plot max, min, avg and max freq of WC model over two parameter ranges. 
    """
    # Default parameter settings
    param_dict = {'exc_ext':[0.75], 'c_excinh':12, 'c_inhexc':12, 'c_inhinh':3} 

    # Naming
    name_dict = {'exc_ext':'Baseline Exc. Drive', 'c_excinh':'E-I Coupling', 'c_inhexc':'I-E Coupling', 'c_inhinh':'I-I Coupling', 
                'sigma_ou':'Noise Level'}

    # Convert values to list if exc_ext is in the input
    for key, value in kwargs.items():
        if key == 'exc_ext': 
            kwargs['exc_ext'] = [list(val) for val in kwargs['exc_ext']]

    # Param range
    p1, p2 = list(kwargs.keys())
    p1_range, p2_range = list(kwargs.values())
    p2_range = p2_range[::-1]

    # Noise Parameters
    nr_noise_levels = 1
    noise_levels = np.linspace(0.00,0.01,nr_noise_levels)

    pdf = PdfPages(filename)
    # Noise loop
    for noise_level in noise_levels:
        # Set up figures
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14,12))
        
        # Init result matrices
        max_mat, min_mat, avg_mat, freq_mat = [np.zeros((len(p1_range),len(p2_range))) for _ in range(4)]

        # Sets up the parameter dictionaries for simulation
        params = []
        for val_1, val_2 in itertools.product(p1_range, p2_range):
            param_dict.update({p1: val_1, p2: val_2, 'sigma_ou':noise_level})
            params.append(param_dict.copy())
            
        # Runs the model in parallel over Gaba parameter 
        with Pool(5) as p:
            result = p.map(evaluate_wc, params)
        max_values, min_values, avg_values, max_freq = list(zip(*result))                    
        
        max_mat.ravel()[:] = max_values
        min_mat.ravel()[:] = min_values
        avg_mat.ravel()[:] = avg_values
        freq_mat.ravel()[:] = max_freq
            
        # Plot heatmap of diff and freq mat
        sns.heatmap(max_mat, ax=axes.ravel()[0], vmin=0, vmax=0.5, linewidths=.1, cmap='YlGnBu', cbar=True)
        sns.heatmap(min_mat, ax=axes.ravel()[1], vmin=0, vmax=0.5, linewidths=.1, cmap='YlGnBu', cbar=True)
        sns.heatmap(avg_mat, ax=axes.ravel()[2], vmin=0, vmax=0.5, linewidths=.1, cmap='YlGnBu', cbar=True)
        sns.heatmap(freq_mat, ax=axes.ravel()[3], vmin=0, vmax=80, center=40, linewidths=.1, cbar=True)

        # Set axes titles and configure axes
        titles = ['Max. Value', 'Min. Value', 'Avg. Value', 'Max. Frequency']
        for ax, title in zip(axes.ravel(), titles): 
            ax.set_title(title)
            ax.set_xticks(np.arange(len(p1_range))+0.5)
            ax.set_yticks(np.arange(len(p2_range))+0.5)
            ax.set_xticklabels([str(np.round(p,1)) for p in p1_range])
            ax.set_yticklabels([str(np.round(p,1)) for p in p2_range])
        
        # Set title
        title = ''
        for key, value in param_dict.items():
            if key not in [p1, p2]:
                if isinstance(value, list): value = value[0]
                title += f' {name_dict[key]}: {value},'
        title = title[:-1]

        # Set common x, y axis labels
        fig.suptitle(title, fontsize=20)
        fig.text(0.5, 0.04, name_dict[p1], ha='center', fontsize=20)
        fig.text(0.04, 0.5, name_dict[p2], va='center', rotation='vertical', fontsize=20)

        # Save to pdf page
        pdf.savefig(fig)
    pdf.close()

def plot_gaba(resolution):
    """
    Plot effect of gaba changes on wc model
    """
    # Gaba Parameters
    ii_couplings = np.linspace(0, 5, resolution)
    ie_couplings = np.linspace(9, 18, resolution)
    plot_2_params(filename='Gaba.pdf', c_inhexc=ie_couplings, c_inhinh=ii_couplings)

def plot_nmda(resolution):
    """
    Plot effect of NMDA changes on wc model
    """
    exc_inputs = np.linspace(0.5, 1, resolution)
    ei_couplings = np.linspace(9, 18, resolution)
    plot_2_params(filename='NMDA', exc_ext=exc_inputs, c_excinh=ei_couplings)

if __name__ == "__main__":
    #plot_4_params()
    plot_gaba(resolution=3)
    #plot_nmda(resolution=2)
