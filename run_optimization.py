from optimization import optimization

if __name__ == "__main__":
    opt = optimization(DataFile='data/Group-FEP_Freq-Alpha_Group-Mean-FC_orth-lowpass-corr.npy', mode='FC', method='corr')
    opt.optimize()
