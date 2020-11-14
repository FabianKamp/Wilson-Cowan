import numpy as np 
def pearson(m1,m2):
    """
    Computes Correlation matrix 
    """
    n = m1.shape[1]

    m1 -= np.mean(m1, axis=-1, keepdims=True)
    std_m1 = np.std(m1, axis=-1, keepdims=True)
    std_m1[np.isclose(std_m1,0)] = 1
    m1 /= std_m1

    m2 -= np.mean(m2, axis=-1, keepdims=True)
    std_m2 = np.std(m2, axis=-1, keepdims=True)
    std_m2[np.isclose(std_m2,0)] = 1
    m2 /= std_m2

    corr = np.matmul(m1,m2.T)/n 
    return corr

def mut_inf(m1,m2):
    from pyinform import utils, mutual_info
    # Bin Timeseries
    bined1,number1,width1 = utils.bin_series(m1, b=20)
    bined2,number2,width2 = utils.bin_series(m2, b=20)
    # Compute mutual info between time courses
    m_info = []
    for tc1,tc2 in zip(bined1,bined2):
        m_info.append(mutual_info(tc1, tc2))
    m_info = np.array(m_info)
    return m_info
