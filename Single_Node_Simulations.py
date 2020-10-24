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

def get_wc_stats(tc, fsample):
    """
    Analysis the time course of the wc-signal
    """
    # Get Outpus stats
    # Get min values
    max_val = np.max(tc)
    min_val = np.min(tc)
    # Get avg value
    avg_value = np.mean(tc)
    # Get max freq greater than 0
    timepoints = len(tc)
    exc_f = fft(tc)
    exc_f = 2.0/timepoints * np.abs(exc_f[0:int(timepoints/2)])
    freqs = np.linspace(0.0, (1.0/2.0)*fsample, int(timepoints/2))
    freqs = freqs[exc_f>1e-2]
    max_freq = freqs[-1]
    stats = [max_val, min_val, avg_value, max_freq]
    return stats

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
    stats = get_wc_stats(exc_tc, fsample)
    return stats

def plot_4_params(resolution):
    """
    Function to plot the effect of 
    """
    # Noise Parameters
    nr_noise_levels = 1
    noise_levels = np.linspace(0.00,0.01,nr_noise_levels)
    # NMDA Parameters
    nr_parameter_levels = resolution
    exc_inputs = np.linspace(0.5, 1, nr_parameter_levels)
    ei_couplings = np.linspace(9, 18, nr_parameter_levels)[::-1]
    # Gaba Parameters
    ii_couplings = np.linspace(0, 5, nr_parameter_levels)[::-1]
    ie_couplings = np.linspace(9, 18, nr_parameter_levels)
    pdf = PdfPages('testing2.pdf')
    
    # Noise loop
    for noise_level in noise_levels:
        # Set up figures
        fig_list = []; axes_list = []; cbar_list = []
        fig_names = ['Maximum Value', 'Minimum Value', 'Average Value', 'Maximum Frequence']       
        for fig_name in fig_names:
            fig, axes = plt.subplots(nr_parameter_levels, nr_parameter_levels, figsize=(15,15))
            fig.suptitle(f'{fig_name}, Noise Level: {noise_level}', fontsize=25)
            fl_ax = axes.ravel()
            cbar = fig.add_axes([.92, 0.15, .03, 0.7])
            fig.text(0.5, 0.04, 'Baseline Excitation', ha='center', fontsize=20)
            fig.text(0.04, 0.5, 'E-I Coupling', va='center', rotation='vertical', fontsize=20)
            fig_list.append(fig); axes_list.append(fl_ax); cbar_list.append(cbar)

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
            
            # Plot heatmap of diff and freq mat
            sns.heatmap(max_mat, ax=axes_list[0][m], vmin=0, vmax=0.5, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else cbar_list[0])
            sns.heatmap(min_mat, ax=axes_list[1][m], vmin=0, vmax=0.5, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else cbar_list[1])
            sns.heatmap(avg_mat, ax=axes_list[2][m], vmin=0, vmax=0.5, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else cbar_list[2])           
            sns.heatmap(freq_mat, ax=axes_list[3][m], center=40, vmin=1, vmax=80, linewidths=.1, cmap="YlGnBu", cbar=m == 0, cbar_ax=None if m else cbar_list[3])
            
            # Configure axes
            for ax in axes_list:            
                ax[m].set_ylabel(f'E-I: {np.round(ei_c, 2)}', labelpad=20, fontsize=15)
                ax[m].set_xlabel(f'Base.: {np.round(e_input,2)}', labelpad=20, fontsize=15)
                ax[m].set_xticks(np.arange(0,nr_parameter_levels)+0.5)
                ax[m].set_yticks(np.arange(0,nr_parameter_levels)+0.5)
                ax[m].set_xticklabels([str(np.round(ie_c,1)) for ie_c in ie_couplings])
                ax[m].set_yticklabels([str(np.round(ii_c,1)) for ii_c in ii_couplings])
        
        # Configure colorbar 
        for cbar in cbar_list:
            cbar.tick_params(labelsize=10, left=False, labelleft=False, right=True, labelright=True)
        
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for axes in axes_list:
            for ax in axes:
                ax.label_outer()

        # Save to pdf page
        for fig in fig_list:
            pdf.savefig(fig)

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
    plot_4_params(10)
    #plot_gaba(resolution=3)
    #plot_nmda(resolution=2)
