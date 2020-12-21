'''
This likelihood ratio classifier code is converted from the original matlab file created by Prof Dr.Ir. RNJ Veldhuis
In this python file, I try to reproduce the same result from the matlab file and probably the result will have a little different with the original
'''
import numpy as np 

np.set_printoptions(precision=4)

#This function is for training the 
def train_LLR(X_tr, Npc, Ndc):
    N1 = X_tr.shape[0]
    N = X_tr.shape[1]

    DimFeat = int((N1-1)/2)

    Npc = min(DimFeat, Npc)
    Ndc = min(Npc,Ndc)

    ixD = np.argwhere(X_tr[-1] == 0).T
    ixS = np.argwhere(X_tr[-1] == 1).T
    Ns = len(ixS[0])
    Nd = len(ixD[0])

    X = np.concatenate((X_tr[0:DimFeat,:], X_tr[DimFeat:2*DimFeat,:]), axis = 1)
    M_global_hat = np.mean(X, axis=1)[:,np.newaxis]
    X = X - M_global_hat
    U_total, S_diag, VH = np.linalg.svd(X, full_matrices=False)
    V_total = np.transpose(VH)
    S_total = np.diagflat(S_diag)

    U_total = U_total[:,:Npc]
    S_total = S_total[:Npc,:Npc]
    V_total = V_total[:,:Npc]

    B_psinv = np.sqrt(2 * N) * np.matmul(np.linalg.inv(S_total), np.transpose(U_total))

    temp_sqrt = np.sqrt(2*N) * np.transpose(
        np.concatenate((V_total[ixS,:].reshape((2400,-1)), 
        V_total[N+ixS,:].reshape((2400,-1)))))

    U_XY1, S_XYk1, V_XYk1 = np.linalg.svd(temp_sqrt, full_matrices=False)
    V_XY = np.transpose(V_XYk1)
    S_XY1 = np.diagflat(S_XYk1)

    V_X = V_XY[:Ns,:]
    V_Y = V_XY[Ns:2*Ns,:]

    temp = np.concatenate((V_Y, V_X))
    mul1 = np.matmul(S_XY1, np.transpose(V_XY))
    mul2 = np.matmul(temp, S_XY1)
    mul = np.matmul(mul1, mul2)
    U_XY2, S_XYk2, _ = np.linalg.svd(mul, full_matrices=False)
    S_XY2 = np.diagflat(S_XYk2)

    U_XY2 = U_XY2[:,:Ndc]
    S_XY2 = S_XY2[:Ndc,:Ndc]

    stdB_hat = np.sqrt(np.diag(S_XY2)[:,np.newaxis]/(2*Ns))

    B_psinv = np.matmul(np.matmul(np.transpose(U_XY2), np.transpose(U_XY1)), 
            B_psinv)

    return M_global_hat, B_psinv, stdB_hat


#This function is used to calculate the likelihood ratio classifier
def LLR_computation(X, M_global_hat, B_psinv, stdB_hat):
    DimFeat = M_global_hat.shape[0]
    DimRed = stdB_hat.shape[0]
    N = X.shape[1]
    varB = stdB_hat**2

    temp1 = np.matmul(B_psinv, (X[:DimFeat,:] - M_global_hat))
    temp2 = np.matmul(B_psinv, (X[DimFeat:2*DimFeat,:] - M_global_hat))
    Y = np.concatenate((temp1,temp2))

    temp3 = np.multiply(-((Y[:DimRed,:] - Y[DimRed:2*DimRed,:])**2), 
                        np.tile((varB / (1-varB)), (1,N)))
    
    temp4 = np.multiply((Y[:DimRed,:] + Y[DimRed:2*DimRed,:])**2, 
                        np.tile((varB / (1+varB)), (1,N)))

    temp5 = 0.25 * np.sum((temp3 + temp4),axis=0)[np.newaxis,:] 
    temp6 = -0.5 * np.sum(np.nan_to_num(np.log(1 - varB**2)),axis=0)

    LLR = temp5 + temp6
    return LLR
