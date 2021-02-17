from optimization import optimization

if __name__ == "__main__":
    opt = optimization(DataFile='data/Group-FEP_Freq-Alpha_Avg-CCD-Hist.npy', mode='CCD', method='ksd')
    opt.optimize()
