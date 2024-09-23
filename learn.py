import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lib.algorithm import IterativeShrinkage
from sklearn.covariance import GraphicalLassoCV


def main():
    #データ読み込み
    data = pd.read_csv("./sparsemodeling-gaussian-graphical-model/csv/decathlon.csv", encoding='utf-8')
    
    #不要なカラム削除/カラム値変更
    data.drop(columns=['Name'], inplace=True)

    #標本分散共分散行列作成
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    cov_S = np.cov(data, rowvar=0, bias=1)
    
    #ライブラリでのグラフィカルラッソ
    model = GraphicalLassoCV()
    model.fit(data)
    matrix = model.precision_
    np.savetxt("./sparsemodeling-gaussian-graphical-model/csv/output2.csv", matrix,delimiter=',')

    #スクラッチでのグラフィカルラッソ
    niter = 51
    lam = 0.16
    iterativeShrinkage = IterativeShrinkage(cov_S, lam)
    x = iterativeShrinkage.GMM_CD(niter=niter)
    np.savetxt("./sparsemodeling-gaussian-graphical-model/csv/output.csv", x,delimiter=',')

if __name__ == "__main__":
    main()