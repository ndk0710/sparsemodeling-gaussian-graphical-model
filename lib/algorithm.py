import numpy as np
import copy

class AlternatingDeterminationMethodOfMultipliers(object):
    """ 交互方向乗数法 """
    def __init__(self, S, lam):
        self.S = S                              # mxmの標本共分散行列S（m：パラメータの次元数10）
        self.eps = 1e-10                        # 更新誤差の閾値
        self.omega = np.full((S.shape[0], S.shape[1]), 0.5)
        self.omega_0 = np.full((S.shape[0], S.shape[1]), 0.5)
        self.U = np.full((S.shape[0], S.shape[1]), 0.5)
        #self.omega = np.ones((S.shape[0], S.shape[1]))                 # 
        #self.omega_0 = np.ones((S.shape[0], S.shape[1]))               # 
        #self.U = np.ones((S.shape[0], S.shape[1]))                     # 
        self.rho = 1
        self.lam = lam 
    
    # 軟閾値作用素
    def soft_threshold(self, x):
        threshold = self.lam
        return np.where(np.abs(x) > threshold, (np.abs(x) - threshold) * np.sign(x), 0)

    def GMM_ADMM(self, niter=50):

        for k in range(niter):
            
            tmp = self.rho * (self.omega_0 - self.U) - self.S

            # 固有値と固有ベクトルを計算
            eig, P = np.linalg.eig(tmp)
            phi = (eig + np.sqrt(eig **2 + 4 * self.rho))/(2*self.rho)
            
            # 対角化行列作成
            phi = np.diag(phi)

            #（Ω）を更新
            self.omega = np.dot(np.dot(np.linalg.inv(P), phi), P)

            e = (self.omega + self.U) * self.rho
            
            # 更新前パラメータ保存（self.old_x）
            self.old_omega_0 = self.omega_0

             #（Ω0）を更新
            self.omega_0 = AlternatingDeterminationMethodOfMultipliers.soft_threshold(self, e)

            self.U = self.U + self.omega -self.omega_0

            # 1エポック目は飛ばす
            if k != 0:
                oo=(self.omega_0 - self.old_omega_0)*(self.omega_0 - self.old_omega_0)
                # 学習終了条件①（パラメータx0,xの更新誤差の確認）
                if np.dot(self.omega_0.flatten()  - self.old_omega_0.flatten() , self.omega_0.flatten()  - self.old_omega_0.flatten() ) < self.eps:
                    return self.omega_0


        return self.omega_0
    

class IterativeShrinkage(object):
    """ 反復縮小アルゴリズム """
    def __init__(self, S, lam):
        self.S = S                              # mxmの標本共分散行列S（m：パラメータの次元数10）
        self.eps = 1e-10                        # 更新誤差の閾値
        self.lam = lam                          # 正則化項（ラッソL1ノルム）のパラメータ
    
    # 軟閾値作用素
    def soft_threshold(self, x):
        threshold = self.lam
        return np.where(np.abs(x) > threshold, (np.abs(x) - threshold) * np.sign(x), 0)
    
    # 列の入替え
    def swap_columns(arr, col1, col2):
        """
        配列の2つの列を入れ替える

        Args:
            arr: NumPy配列
            col1: 入れ替える列1のインデックス
            col2: 入れ替える列2のインデックス

        Returns:
            列を入れ替えたNumPy配列
        """

        tmp = copy.deepcopy(arr[:, col1])
        arr[:, col1] = arr[:, col2]
        arr[:, col2] = tmp

        return arr
    
    # 行の入替え
    def swap_rows(arr, row1, row2):
        """
        配列の2つの行を入れ替える

        Args:
            arr: NumPy配列
            row1: 入れ替える行1のインデックス
            row2: 入れ替える行2のインデックス

        Returns:
            行を入れ替えたNumPy配列
        """

        tmp = copy.deepcopy(arr[row1, :])
        arr[row1, :] = arr[row2, :]
        arr[row2, :] = tmp

        return arr


    def GMM_CD(self, niter=50):
        # 初期化（sigma）
        self.sigma = self.S + self.lam * np.identity(self.S.shape[0])
        # 初期化（omega）
        self.omega = np.linalg.inv(self.sigma)
        
        # パラメータの次元数だけ処理
        for index in range(self.S.shape[0]):
            #1 self.omegaのj行目とp行目、j列目とp列目を入替える
            self.omega = IterativeShrinkage.swap_columns(self.omega, index, self.S.shape[0] - 1)
            self.omega = IterativeShrinkage.swap_rows(self.omega, index, self.S.shape[0] - 1)

            self.sigma = IterativeShrinkage.swap_columns(self.sigma, index, self.S.shape[0] - 1)
            self.sigma = IterativeShrinkage.swap_rows(self.sigma, index, self.S.shape[0] - 1)

            self.S = IterativeShrinkage.swap_columns(self.S, index, self.S.shape[0] - 1)
            self.S = IterativeShrinkage.swap_rows(self.S, index, self.S.shape[0] - 1)
            
            self.tilde_sigma = self.sigma[0:self.S.shape[0]-1, 0:self.S.shape[0]-1]
            self.vector_sigma = self.sigma[0:self.S.shape[0]-1, -1]
            self.scalar_sigma = self.sigma[-1, -1]
            self.vector_S = self.S[0:self.S.shape[0]-1, -1]

            self.beta_0 = np.dot(np.linalg.inv(self.tilde_sigma), self.vector_sigma)

            tilde_sigma_diag = np.diag(self.tilde_sigma)
            
            #2 Lasso回帰でself.beta_0を更新
            for k in range(niter):
                for dim in range(len(self.vector_S)):

                    calc = np.dot(self.tilde_sigma,self.beta_0)
                    A = self.vector_S[dim] - (calc[dim] - tilde_sigma_diag[dim] * self.beta_0[dim])
                    e = IterativeShrinkage.soft_threshold(self, A)
                    self.beta_0[dim] = e / tilde_sigma_diag[dim]

                # 1エポック目は飛ばす
                if k != 0:
                    oo=np.dot(self.beta_0  - self.old_beta_0, self.beta_0  - self.old_beta_0)
                    # 学習終了条件①（パラメータx0,xの更新誤差の確認）
                    if np.dot(self.beta_0  - self.old_beta_0, self.beta_0  - self.old_beta_0) < self.eps:
                        break

                self.old_beta_0 = copy.deepcopy(self.beta_0)

            #3 self.vector_sigmaとscalar_sigmaとvector_omegaを更新
            self.vector_sigma = np.dot(self.tilde_sigma, self.beta_0)
            scalar_omega = 1/(self.scalar_sigma - np.dot(self.vector_sigma.T, self.beta_0))
            vector_omega = -1 * scalar_omega * self.beta_0

            #4 self.omegaとself.sigmaを更新
            self.sigma[0:self.S.shape[0]-1, -1] = self.vector_sigma
            self.sigma[-1, 0:self.S.shape[0]-1] = self.vector_sigma.T

            self.omega[0:self.S.shape[0]-1, -1] = vector_omega
            self.omega[-1, 0:self.S.shape[0]-1] = vector_omega.T
            self.omega[-1, -1] = scalar_omega

            #5 self.omegaのj行目とp行目、j列目とp列目を元に戻す
            self.omega = IterativeShrinkage.swap_columns(self.omega, self.S.shape[0] - 1, index)
            self.omega = IterativeShrinkage.swap_rows(self.omega, self.S.shape[0] - 1, index)

            self.sigma = IterativeShrinkage.swap_columns(self.sigma, self.S.shape[0] - 1 ,index)
            self.sigma = IterativeShrinkage.swap_rows(self.sigma, self.S.shape[0] - 1, index)

            self.S = IterativeShrinkage.swap_columns(self.S, self.S.shape[0] - 1, index)
            self.S = IterativeShrinkage.swap_rows(self.S, self.S.shape[0] - 1, index)
        
        return self.omega