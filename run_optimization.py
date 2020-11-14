from optimization import optimization

if __name__ == "__main__":
    opt = optimization(DataFile='Group-Control_Freq-Alpha_Group-Mean-FC_orth-lowpass-corr.npy')
    opt.optimize()
