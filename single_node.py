import numpy as np 
from neurolib.models.wc import WCModel
import seaborn as sns
from itertools import product
from multiprocessing import Pool
from scipy.fft import fft
from time import time 
import pandas as pd

def get_spectrum(tc, fsample):
    """
    Compute frequency spectrum using scipy.fft
    """
    timepoints = len(tc)
    spect = fft(tc)
    spect[0] = 0
    spect = 2.0/timepoints * np.abs(spect[0:int(timepoints/2)])
    freqs = np.linspace(0.0, (1.0/2.0)*fsample, int(timepoints/2))    
    return spect, freqs

def get_wc_stats(tc, fsample):
    """
    Analysis the time course of the wc-signal
    """
    # Get Outpus stats
    # Get min/max values
    max_val = np.max(tc)
    min_val = np.min(tc)
    # Get avg value
    diff = max_val - min_val
    # Get max freq greater than 0
    spect, freqs = get_spectrum(tc, fsample)
    max_freq = freqs[np.argmax(spect)]
    gamma = np.sum(spect[(freqs>=30) & (freqs<=46)])
    stats = {'Max': max_val, 'Min': min_val, 'Diff': diff, 'Gamma': gamma, 'Max_Freq': max_freq}
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
    time = 2.*1000
    fsample = 10000.
    wc.params['duration'] = time
    # Run model with parameter setting
    for key, value in params.items():
        wc.params[key] = value
    wc.run()
    exc_tc = wc.outputs.exc[0,100:]
    inh_tc = wc.outputs.inh[0,100:]
    stats = get_wc_stats(exc_tc, fsample)
    return stats

def f_gaba(param): 
    """
    Function to translate f_gaba factor to WC-parameter space
    """
    c_ie_range = [8, 16]
    c_ii_range = [0, 6]
    c_ie = np.min(c_ie_range) + float(np.diff(c_ie_range)) * param
    c_ii = np.min(c_ii_range) + float(np.diff(c_ii_range)) * param
    return c_ie, c_ii

def f_nmda(param):
    """
    Function to translate f_nmda factor to WC-parameter space
    """
    #p_e_range = [0.45, 0.9]
    #p_i_range = [0, 0.01]
    c_ei_range = [5, 20]
    #p_e = np.min(p_e_range) + float(np.diff(p_e_range)) * param
    p_e = 0.5
    #p_i = np.min(p_i_range) + float(np.diff(p_i_range)) * param
    c_ei = np.min(c_ei_range) + float(np.diff(c_ei_range)) * param
    return p_e, c_ei 

def run_wc():
    """
    Runs WC model in parallel
    """
    #NMDA
    p_e_range = np.arange(0.1, 3, 0.05)
    params=[]
    for p_e in p_e_range:  
        c_ie, c_ii = f_gaba(0.5)
        _, c_ei = f_nmda(0.5)      
        params.append({'exc_ext':np.array([p_e]), 'c_excinh':c_ei,
        'c_inhexc':12.5, 'c_inhinh':3})
    with Pool(25) as p:
        results = p.map(evaluate_wc, params)
    para_df = pd.DataFrame(params)
    df = pd.DataFrame(results)
    df = df.join(para_df)    
    df.to_pickle('results/single_node_p.pkl')

    # f gaba/nmda
    param_range = np.linspace(0,1,100)
    params = []
    f_list = []
    for gaba,nmda in product(param_range[::-1], param_range): 
        c_ie, c_ii = f_gaba(gaba)
        p_e, c_ei = f_nmda(nmda)        
        params.append({'exc_ext':np.array([p_e]), 'c_excinh':c_ei, 'c_inhexc':c_ie, 'c_inhinh':c_ii})
        f_list.append({'f_gaba':gaba,'f_nmda':nmda})
    with Pool(25) as p:
        results = p.map(evaluate_wc, params)    
    para_df = pd.DataFrame(f_list)
    df = pd.DataFrame(results)
    df = df.join(para_df)
    df.to_pickle('results/single_node_f.pkl')

if __name__ == "__main__":
    start = time()
    run_wc()
    end = time()
    print('Time: ', end-start)

