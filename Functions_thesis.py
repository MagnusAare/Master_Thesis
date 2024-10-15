# from Functions_thesis import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import shap
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from Functions_plotting import *
import warnings
import random
import math
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from matplotlib import colors as mcolors
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from collections import defaultdict
import seaborn as sns
sns.set_palette("tab10")
plt.rcParams.update({
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 12,     # X and Y label size
    'xtick.labelsize': 12,    # X tick labels size
    'ytick.labelsize': 12,    # Y tick labels size
    'legend.fontsize': 12,    # Legend font size
    'legend.title_fontsize': 12  # Legend title font size
})
# from sklearn_extra.cluster import KMedoids

# from Functions_plotting import plot_cluster

# from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

########## CLASSES ##########


class Traders_Dual:
    """Class to configurate the Traders/producers problem with Dual formulation. - This is not use in the final project"""

    def __init__(self, Problem, T, problem_parameters, n_var, n_constraints, variables, Target):
        self.Problem = Problem
        self.T = T
        self.LT = len(T)
        self.P_DIS_CH = problem_parameters['P_DIS_CH'] / \
            problem_parameters['PW_cap']
        self.eff = problem_parameters['eff']
        self.SOC_init = problem_parameters['SOC_init'] / \
            problem_parameters['PW_cap']
        self.SOC_cap = problem_parameters['SOC_cap'] / \
            problem_parameters['PW_cap']
        self.PW_cap = problem_parameters['PW_cap']/problem_parameters['PW_cap']
        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.Target = Target
        self.b = self.Vector_b()
        self.w_len = self.n_var*self.LT+1

        # Placeholders:
        self.A_train = None
        self.A_test = None
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None

    def Matrix_A(self, DA_price_mean):
        """Creates the A matrix of the problem"""
        LT = self.LT
        ID = np.identity(LT)
        M0 = np.zeros((LT, LT))
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), 1)
        V0 = np.zeros(LT)

        Lambda_col = np.array(DA_price_mean).reshape(-1, 1)

        V0_col = V0.reshape(-1, 1)

        A = [[1, V0, V0, V0, V0, V0],
             [-1, V0, V0, V0, V0, V0],
             [-Lambda_col, ID, M0, M0, M0, M0],
             [V0_col, ID, ID, M0, M0, -self.eff*ID],
             [-Lambda_col, M0, M0, ID, M0, (1/self.eff)*ID],
             [V0_col, M0, M0, M0, ID, M],
             [1, V0, V0, V0, V0, V0],
             [V0_col, ID, M0, M0, M0, M0],
             [V0_col, M0, ID, M0, M0, M0],
             [V0_col, M0, M0, ID, M0, M0],
             [V0_col, M0, M0, M0, ID, M0]]
        A = np.block(A)
        # self.A = A
        return A

    def Vector_b(self):
        """Creates the b vector of the problem"""
        # b
        LT = self.LT
        b = np.zeros(self.n_constraints*LT+3)
        b[0] = 1
        b[1] = -1

        return b

    def Vector_c(self, data_sample):
        """Creates the c vector of the problem"""
        LT = self.LT
        c = np.zeros(self.n_var*LT)
        c[:LT] = data_sample[self.Target[1]]
        c[LT:2*LT] = self.P_DIS_CH
        c[2*LT:3*LT] = self.P_DIS_CH
        c[3*LT:4*LT] = self.SOC_cap
        c[4*LT] = self.SOC_init
        c = np.insert(c, 0, 0)

        return c

    def dataset_creator(self, data: pd.DataFrame, IDS: list, features: list):
        """Specific dataset creator for traders on dual formulation. This is needed because of A"""
        self.features = features
        self.dataset = {}
        for ids in IDS:
            temp = data[data['ID'] == ids]
            self.dataset[ids] = {col: temp[col].to_numpy()
                                 for col in temp.columns}
            self.dataset[ids]['cost_vector'] = self.Vector_c(temp)
            self.dataset[ids]['A'] = self.Matrix_A(self.dataset[ids]['DA_DK2'])
            self.dataset[ids]['z'], self.dataset[ids]['w'], self.dataset[ids]['constraints_dual'], _ = Oracle(
                self.dataset[ids]['cost_vector'], self, self.dataset[ids]['A'])
        return self.dataset


class Traders_Lagrangian:
    """Class to configure the traders/producers problem where lagrangian relaxation to reformulated the problem into SPO format"""

    def __init__(self, Problem, T, problem_parameters, n_var, n_constraints, variables, Target):
        self.Problem = Problem
        self.T = T
        self.P_DIS_CH = problem_parameters['P_DIS_CH'] / \
            problem_parameters['PW_cap']
        self.eff = problem_parameters['eff']
        self.SOC_init = problem_parameters['SOC_init'] / \
            problem_parameters['PW_cap']
        self.SOC_cap = problem_parameters['SOC_cap'] / \
            problem_parameters['PW_cap']
        self.PW_cap = problem_parameters['PW_cap']/problem_parameters['PW_cap']
        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.Target = Target
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.soc_init[0] = self.SOC_init
        self.w_len = self.n_var*self.LT
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None
        self.B = None

    def Vector_A_b(self):
        "Creates the A and b for the problem"
        LT = self.LT
        eff = self.eff

        ID = np.identity(LT)
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), -1)
        M0 = np.zeros((LT, LT))

        row_blocks = [[M0, -ID, M0, M0, M0],
                      [M0, M0, -ID, M0, M0],
                      [M0, M0, M0, -ID, M0],
                      [M0, -eff*ID, (1/eff)*ID, M, M0],
                      [M0, eff*ID, -(1/eff)*ID, -M, M0],
                      [ID, M0, M0, M0, M0],
                      [M0, ID, M0, M0, M0],
                      [M0, M0, ID, M0, M0],
                      [M0, M0, M0, ID, M0],
                      [M0, M0, M0, M0, ID],
                      [M0, M0, M0, M0, -ID]]
        A = np.block(row_blocks)

        b = np.zeros((LT*self.n_constraints))
        b[self.T] = -self.P_DIS_CH
        b[LT:LT*2] = -self.P_DIS_CH
        b[2*LT:LT*3] = -self.SOC_cap
        b[3*LT:LT*4] = self.SOC_init
        b[4*LT:LT*5] = -self.SOC_init
        b[-LT*2:-LT] = 1
        b[-LT:] = -1
        return A, b

    def Vector_c(self, DA, mu_res, PW):
        """Creates the c vector of the problem"""
        LT = self.LT
        c = np.zeros((LT*self.n_var))
        c[:LT] = -DA+mu_res
        c[LT:LT*2] = mu_res
        c[2*LT:LT*3] = -DA
        # 0
        c[-LT:] = -mu_res*PW
        return c

    def w_vector(self, sample_dict):
        """Creates the w vector of the problem"""
        LT = self.LT
        w = np.zeros((LT*self.n_var))
        w[:LT] = sample_dict['pW']
        w[LT:2*LT] = sample_dict['pCH']
        w[2*LT:3*LT] = sample_dict['pDIS']
        w[3*LT:4*LT] = sample_dict['SOC']
        w[-LT:] = 1
        return w

    def dataset_creator(self, data, IDS, features):
        """Specific dataset creator for traders with lagrangian formulation. This is needed because of mu"""
        self.features = features
        self.dataset = {}
        for ids in IDS:
            temp = data[data['ID'] == ids]
            PW = temp['windpower']
            DA = temp['DA_DK2']
            self.dataset[ids] = producers_problem_standard_form(
                self, DA, PW)
            self.dataset[ids].update({col: temp[col].to_numpy()
                                      for col in temp.columns})
            mu = self.dataset[ids]['mu']
            self.dataset[ids]['cost_vector'] = self.Vector_c(DA, mu, PW)
            # sample_w = producers_problem_lagrangian_relaxation(
            # self, DA, PW, mu)
            # self.dataset[ids]['w'] = self.w_vector(sample_w) #Relaxed w
            self.dataset[ids]['w'] = self.w_vector(self.dataset[ids])  # True w
        return self.dataset

    def predict(self, x_test):
        """Predict the SPO prediction given x_test, when B is found."""
        if self.B is None:
            raise ValueError("B is not loaded. Please fit and load B first")
        else:
            return x_test @ self.B.T


class EDP:
    """Class to configure the economic dispatch problem. This can configure: multihour copperplate, multihour OPF"""

    def __init__(self, Problem, G, T, N, NN, NG, WN, parameters, variables, Target):
        self.Problem = Problem
        self.T = T
        self.G = G
        self.N = N
        self.NN = NN
        self.WN = WN
        self.TransmissionLines, self.L, self.LL = self.Tlines()
        self.NG = NG
        self.cost_G = parameters['cost_G']
        self.P_G = parameters['P_G']
        self.P_F = parameters['P_F']
        self.sus = parameters['sus']
        self.P_W = parameters['P_W']
        self.GN = parameters['GN']
        self.DN = parameters['DN']
        self.DemandFractions = parameters['DemandFractions']
        self.variables = variables
        self.Target = Target
        self.LT = len(T)
        self.LG = len(G)
        self.LN = len(N)
        self.w_len = self.LT * self.LN + self.LT * self.LG + \
            self.LT * self.LL + (self.LT if self.LN > 1 else 0)
        self.b_len = 2*self.LT*self.LG + \
            (2*self.LT*self.LN if self.LN > 1 else 0) + self.LT*self.LL
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.dual_train = None
        self.dual_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None

    def Tlines(self):
        """Connects nodes N to M with transmission lines"""
        TransmissionLines = []
        for n in self.NN:
            for m in self.NN[n]:
                TransmissionLines.append([n, m])
        LL = len(TransmissionLines)
        L = range(LL)

        return TransmissionLines, L, LL

    def Vector_A_b(self):
        """Creates A and b for the problem"""
        b = np.zeros(self.b_len)
        A = np.zeros((self.b_len, self.w_len))

        for t in self.T:
            b[t*self.LG:(t+1)*self.LG] = - self.cost_G

        for t in self.T:
            for g in self.G:
                A[t*self.LG+g, self.NG[g]*self.LT+t] = -1
                A[t*self.LG+g, self.LN*self.LT+t*self.LG+g] = 1
                if self.LN > 1:
                    A[self.LT*self.LG+2*self.LN*self.LT+t *
                        self.LG+g, self.LN*self.LT+t*self.LG+g] = 1
                else:
                    A[self.LT*self.LG+t *
                        self.LG+g, self.LN*self.LT+t*self.LG+g] = 1
        if self.LN > 1:
            for l in self.L:
                for t in self.T:
                    n = self.TransmissionLines[l][0]
                    m = self.TransmissionLines[l][1]
                    A[self.LT*self.LG+n*self.LT+t, n*self.LT+t] += -self.sus[n, m]
                    A[self.LT*self.LG+n*self.LT+t, m*self.LT+t] += self.sus[m, n]
                    A[self.LT*self.LG+n*self.LT+t, self.LN*self.LT +
                        self.LT*self.LG+l*self.LT+t] += -self.sus[n, m]
                    A[self.LT*self.LG+m*self.LT+t, self.LN*self.LT +
                        self.LT*self.LG+l*self.LT+t] += self.sus[m, n]
                    A[self.LT*self.LG+self.LN*self.LT+n *
                        self.LT+t, n*self.LT+t] += self.sus[n, m]
                    A[self.LT*self.LG+self.LN*self.LT+n *
                        self.LT+t, m*self.LT+t] += -self.sus[m, n]
                    A[self.LT*self.LG+self.LN*self.LT+n*self.LT+t, self.LN *
                        self.LT+self.LT*self.LG+l*self.LT+t] += self.sus[n, m]
                    A[self.LT*self.LG+self.LN*self.LT+m*self.LT+t, self.LN *
                        self.LT+self.LT*self.LG+l*self.LT+t] += -self.sus[m, n]
                    A[self.LT*self.LG+2*self.LN*self.LT+self.LT*self.LG+l *
                        self.LT+t, self.LN*self.LT+self.LT*self.LG+l*self.LT+t] = 1
                    A[self.LT*self.LG+0*self.LT+t, self.LN*self.LT +
                        self.LT*self.LG+self.LL*self.LT+t] = 1
                    A[self.LT*self.LG+self.LN*self.LT+0*self.LT+t, self.LN *
                        self.LT+self.LT*self.LG+self.LL*self.LT+t] = -1

        return A, b

    def Vector_c(self, data_sample):
        """Creates the c vector for the problem"""
        c = np.zeros(self.w_len)

        for n in self.N:
            name = "P_D_N" + str(n)
            c[n*self.LT:(n+1)*self.LT] = - data_sample[name]
        for t in self.T:
            c[self.LN*self.LT+t*self.LG:self.LN *
                self.LT+(t+1)*self.LG] = self.P_G
        for l in self.L:
            c[self.LN*self.LT+self.LT*self.LG+l*self.LT:self.LN*self.LT+self.LT*self.LG +
                (l+1)*self.LT] = self.P_F[self.TransmissionLines[l][0], self.TransmissionLines[l][1]]

        return c

    def predict(self, x_test):
        """Predict the SPO prediction given x_test, when B is found."""
        if self.B is None:
            raise ValueError("B is not loaded. Please fit and load B first")
        else:
            return x_test @ self.B.T


class Traders_equality_original:
    """Class to configure the traders/producers problem where the windpower balance is an equality constraint"""

    def __init__(self, Problem, T, problem_parameters, n_var, n_constraints, variables, Target):
        self.Problem = Problem
        self.T = T
        self.P_DIS_CH = problem_parameters['P_DIS_CH'] / \
            problem_parameters['PW_cap']
        self.eff = problem_parameters['eff']
        self.SOC_init = problem_parameters['SOC_init'] / \
            problem_parameters['PW_cap']
        self.SOC_cap = problem_parameters['SOC_cap'] / \
            problem_parameters['PW_cap']
        self.PW_cap = problem_parameters['PW_cap']/problem_parameters['PW_cap']

        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.Target = Target
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.soc_init[0] = self.SOC_init
        self.w_len = self.n_var*self.LT
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None
        self.B = None

    def Vector_A_b(self):
        "Returns the A and b for the problem"
        LT = self.LT
        eff = self.eff
        n_constraints = self.n_constraints
        P_DIS_CH = self.P_DIS_CH
        SOC_cap = self.SOC_cap
        soc_init = self.soc_init

        ID = np.identity(LT)
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), -1)
        M0 = np.zeros((LT, LT))

        row_blocks = [[-ID, M0, M0, M0],
                      [M0, -ID, M0, M0],
                      [M0, M0, M0, -ID],
                      [-eff*ID, (1/eff)*ID, M0, M],
                      [eff*ID, -(1/eff)*ID, M0, -M],
                      [ID, M0, M0, M0],
                      [M0, ID, M0, M0],
                      [M0, M0, -ID, M0],
                      [M0, M0, ID, M0],
                      [M0, M0, M0, ID]]

        A = np.block(row_blocks)

        b = np.zeros((LT*n_constraints))
        b[self.T] = -P_DIS_CH
        b[LT:LT*2] = -P_DIS_CH
        b[2*LT:LT*3] = -SOC_cap
        b[3*LT:LT*4] = soc_init
        b[4*LT:LT*5] = -soc_init
        b[-LT*3:-2*LT] = -1
        b[-2*LT:-LT] = 1
        return A, b

    def Vector_c(self, data_sample):
        """Creates the c vector for the problem"""
        LT = self.LT
        c = np.zeros(self.n_var*LT)
        c[:LT] = data_sample[self.Target[0]]
        c[LT:LT*2] = -data_sample[self.Target[0]]
        c[2*LT:LT*3] = -data_sample[self.Target[0]]*data_sample[self.Target[1]]
        return c

    def predict(self, x_test):
        """Predict the SPO prediction given x_test, when B is found."""
        if self.B is None:
            raise ValueError("B is not loaded. Please fit and load B first")
        else:
            return x_test @ self.B.T


class Traders_equality:
    """Class to configure the traders/producers problem where the windpower balance is an equality constraint. Here for when """

    def __init__(self, Problem, T, problem_parameters, n_var, n_constraints, variables, Target):
        self.Problem = Problem
        self.T = T
        self.P_DIS_CH = problem_parameters['P_DIS_CH'] / \
            problem_parameters['PW_cap']
        self.eff = problem_parameters['eff']
        self.SOC_init = problem_parameters['SOC_init'] / \
            problem_parameters['PW_cap']
        self.SOC_cap = problem_parameters['SOC_cap'] / \
            problem_parameters['PW_cap']
        self.PW_cap = problem_parameters['PW_cap']/problem_parameters['PW_cap']

        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.Target = Target
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.soc_init[0] = self.SOC_init
        self.w_len = self.n_var*self.LT
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None
        self.B = None

    def Vector_A_b(self):
        "Returns the A and b for the problem"
        LT = self.LT
        eff = self.eff
        n_constraints = self.n_constraints
        P_DIS_CH = self.P_DIS_CH
        SOC_cap = self.SOC_cap
        soc_init = self.soc_init

        ID = np.identity(LT)
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), -1)
        M0 = np.zeros((LT, LT))

        row_blocks = [[-ID, M0, M0, M0],
                      [M0, -ID, M0, M0],
                      [M0, M0, M0, -ID],
                      [-eff*ID, (1/eff)*ID, M0, M],
                      [eff*ID, -(1/eff)*ID, M0, -M],
                      [ID, M0, M0, M0],
                      [M0, ID, M0, M0],
                      [M0, M0, -ID, M0],
                      [M0, M0, ID, M0],
                      [M0, M0, M0, ID]]

        A = np.block(row_blocks)

        b = np.zeros((LT*n_constraints))
        b[self.T] = -P_DIS_CH
        b[LT:LT*2] = -P_DIS_CH
        b[2*LT:LT*3] = -SOC_cap
        b[3*LT:LT*4] = soc_init
        b[4*LT:LT*5] = -soc_init
        b[-LT*3:-2*LT] = -1
        b[-2*LT:-LT] = 1
        return A, b

    def Vector_c(self, data_sample):
        """Creates c vector for the problem"""
        LT = self.LT
        c = np.zeros(self.n_var*LT)
        c[:LT] = data_sample['DA_DK2']
        c[LT:LT*2] = -data_sample['DA_DK2']
        c[2*LT:LT*3] = -data_sample['DA_DK2']*data_sample["windpower"]
        return c

    def w_vector(self, sample_dict):
        """Creates the w vector, with aux variable y."""
        LT = self.LT
        w = np.zeros((LT*self.n_var))
        w[:LT] = sample_dict['pCH']
        w[LT:2*LT] = sample_dict['pDIS']
        w[2*LT:3*LT] = 1
        w[3*LT:4*LT] = sample_dict['SOC']
        return w

    def dataset_creator(self, data, IDS, features):
        """Specific data set creator because of w"""
        self.features = features
        self.dataset = {}
        for ids in IDS:
            temp = data[data['ID'] == ids]
            PW = temp['windpower']
            DA = temp['DA_DK2']
            sample_res = producers_problem_standard_form_equality(
                self, DA, PW)
            self.dataset[ids] = sample_res
            self.dataset[ids].update({col: temp[col].to_numpy()
                                      for col in temp.columns})
            # mu = self.dataset[ids]['mu']
            self.dataset[ids]['cost_vector'] = self.Vector_c(sample_res)
            self.dataset[ids]['w'] = self.w_vector(self.dataset[ids])  # True w
        return self.dataset

    def predict(self, x_test):
        """Predict the SPO prediction given x_test, when B is found."""
        if self.B is None:
            raise ValueError("B is not loaded. Please fit and load B first")
        else:
            return x_test @ self.B.T


class WF_DA_RT:
    """Class to configurate the Windfarm DA and RT problem"""

    def __init__(self, Problem, T, PW_cap, n_var, n_constraints, variables):
        self.Problem = Problem
        self.T = T
        self.PW_cap = PW_cap
        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.w_len = self.n_var*self.LT
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None

    def Vector_A_b(self):
        LT = self.LT
        n_var = self.n_var
        ID = np.identity(n_var*LT)

        A = np.vstack((-ID, ID))

        b1 = np.ones((n_var*LT))  # PW cap og 1 for aux
        b2 = np.ones((n_var*LT))  # 0 og 1 for aux
        b2[:LT] = 0
        b = np.concatenate((-b1, b2))

        return A, b

    def Vector_c(self, data_sample):
        LT = self.LT
        n_var = self.n_var
        c = np.zeros(n_var*LT)
        c[:LT] = -data_sample['DA_DK2']+data_sample['RT_DK2']
        c[LT:] = -data_sample['RT_DK2']*data_sample['PW']

        return c


class BESS_DA_RT:
    """Class to configurate the BESS DA and RT problem"""

    def __init__(self, Problem, T, P_DIS_CH, eff, SOC_init, SOC_cap, n_var, n_constraints, variables):
        self.Problem = Problem
        self.T = T
        self.P_DIS_CH = P_DIS_CH
        self.eff = eff
        self.SOC_init = SOC_init
        self.SOC_cap = SOC_cap
        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.soc_init[0] = self.SOC_init
        self.w_len = self.n_var*self.LT
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None

    def Vector_A_b(self):
        "Returns the A and b for the problem"
        LT = self.LT
        eff = self.eff
        n_constraints = self.n_constraints
        P_DIS_CH = self.P_DIS_CH
        SOC_cap = self.SOC_cap
        soc_init = self.soc_init

        ID = np.identity(LT)
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), -1)
        M0 = np.zeros((LT, LT))

        row_blocks = [[-ID, M0, -ID, M0, M0],
                      [M0, -ID, M0, -ID, M0],
                      [(1/eff)*ID, -eff*ID, (1/eff)*ID, -eff*ID, M],
                      [-(1/eff)*ID, eff*ID, -(1/eff)*ID, eff*ID, -M],
                      [M0, M0, M0, M0, -ID],
                      [ID, M0, M0, M0, M0],
                      [M0, ID, M0, M0, M0],
                      [M0, M0, ID, M0, M0],
                      [M0, M0, M0, ID, M0],
                      [M0, M0, M0, M0, ID]]

        A = np.block(row_blocks)

        b = np.zeros((LT*n_constraints))
        b[self.T] = -P_DIS_CH
        b[LT:LT*2] = -P_DIS_CH
        b[2*LT:LT*3] = soc_init
        b[3*LT:LT*4] = -soc_init  # Mangler - fordi ellers ser py som -0
        b[4*LT:LT*5] = -SOC_cap
        return A, b

    def Vector_c(self, data_sample):
        LT = self.LT
        c = np.zeros(self.n_var*LT)
        c[:LT] = -data_sample['DA_DK2']
        c[LT:LT*2] = data_sample['DA_DK2']
        c[2*LT:LT*3] = -data_sample['RT_DK2']
        c[3*LT:LT*4] = data_sample['RT_DK2']
        return c


class BESS_DA:
    """Class to configurate the BESS DA"""

    def __init__(self, Problem, T, P_DIS_CH, eff, SOC_init, SOC_cap, n_var, n_constraints, variables):
        self.Problem = Problem
        self.T = T
        self.P_DIS_CH = P_DIS_CH
        self.eff = eff
        self.SOC_init = SOC_init
        self.SOC_cap = SOC_cap
        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.soc_init[0] = self.SOC_init
        self.w_len = self.n_var*self.LT
        self.A, self.b = self.Vector_A_b()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None

    def Vector_A_b(self):
        "Returns the A and b for the problem"
        LT = self.LT
        eff = self.eff
        n_constraints = self.n_constraints
        P_DIS_CH = self.P_DIS_CH
        SOC_cap = self.SOC_cap
        soc_init = self.soc_init

        ID = np.identity(LT)
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), -1)
        M0 = np.zeros((LT, LT))

        row_blocks = [[-ID, M0, M0],
                      [M0, -ID, M0],
                      [(1/eff)*ID, -eff*ID, M],
                      [-(1/eff)*ID, eff*ID, -M],
                      [M0, M0, -ID],
                      [ID, M0, M0],
                      [M0, ID, M0],
                      [M0, M0, ID]]

        A = np.block(row_blocks)

        b = np.zeros((LT*n_constraints))
        b[self.T] = -P_DIS_CH
        b[LT:LT*2] = -P_DIS_CH
        b[2*LT:LT*3] = soc_init
        b[3*LT:LT*4] = -soc_init  # Mangler - fordi ellers ser py som -0
        b[4*LT:LT*5] = -SOC_cap
        return A, b

    def Vector_c(self, data_sample):
        LT = self.LT
        c = np.zeros(self.n_var*LT)
        c[:LT] = -data_sample['DA_DK2']
        c[LT:LT*2] = data_sample['DA_DK2']
        return c


class Traders_equality_one_uncertainty:
    """Class to configure the traders/producers problem where the windpower is known/predicted from sequential"""

    def __init__(self, Problem, T, problem_parameters, n_var, n_constraints, variables, Target):
        self.Problem = Problem
        self.T = T
        self.P_DIS_CH = problem_parameters['P_DIS_CH'] / \
            problem_parameters['PW_cap']
        self.eff = problem_parameters['eff']
        self.SOC_init = problem_parameters['SOC_init'] / \
            problem_parameters['PW_cap']
        self.SOC_cap = problem_parameters['SOC_cap'] / \
            problem_parameters['PW_cap']
        self.PW_cap = problem_parameters['PW_cap']/problem_parameters['PW_cap']

        self.n_var = n_var
        self.n_constraints = n_constraints
        self.variables = variables
        self.Target = Target
        self.LT = len(T)
        self.soc_init = np.zeros((self.LT))
        self.soc_init[0] = self.SOC_init
        self.w_len = self.n_var*self.LT
        self.A = self.Vector_A()

        # Placeholders for later data:
        self.x_train = None
        self.x_test = None
        self.c_train = None
        self.c_test = None
        self.w_train = None
        self.w_test = None
        self.z_train = None
        self.z_test = None
        self.n_samples_train = None
        self.n_samples_test = None
        self.B = None

    def Vector_A(self):
        "Returns the A for the problem"
        LT = self.LT
        eff = self.eff
        ID = np.identity(LT)
        M = np.identity(LT)+np.diag(np.ones(LT-1)*(-1), -1)
        M0 = np.zeros((LT, LT))

        row_blocks = [[-ID, M0, M0, M0],
                      [M0, -ID, M0, M0],
                      [-ID, M0, -ID, M0],
                      [ID, M0, ID, M0],
                      [M0, M0, M0, -ID],
                      [-eff*ID, (1/eff)*ID, M0, M],
                      [eff*ID, -(1/eff)*ID, M0, -M],
                      [ID, M0, M0, M0],
                      [M0, ID, M0, M0],
                      [M0, M0, ID, M0],
                      [M0, M0, M0, ID]]

        A = np.block(row_blocks)
        return A

    def Vector_b(self, PW):
        """Return b for the problem. Here PW is either a prediction or known."""
        LT = self.LT
        n_constraints = self.n_constraints
        P_DIS_CH = self.P_DIS_CH
        SOC_cap = self.SOC_cap
        soc_init = self.soc_init

        b = np.zeros((LT*n_constraints))
        b[:LT] = -P_DIS_CH
        b[LT:LT*2] = -P_DIS_CH
        b[2*LT:LT*3] = -PW
        b[3*LT:LT*4] = PW
        b[4*LT:LT*5] = -SOC_cap
        b[5*LT:LT*6] = soc_init
        b[6*LT:LT*7] = -soc_init
        return b

    def Vector_c(self, data_sample):
        """Creates the c vector for the problem"""
        LT = self.LT
        c = np.zeros(self.n_var*LT)
        c[LT:2*LT] = -data_sample['DA_DK2']
        c[2*LT:LT*3] = -data_sample['DA_DK2']
        return c

    def dataset_creator(self, data, IDS, features):
        """Specific dataset creator for the problem. Needed because b is known."""
        self.features = features
        self.dataset = {}
        for ids in IDS:
            temp = data[data['ID'] == ids]
            self.dataset[ids] = {col: temp[col].to_numpy()
                                 for col in temp.columns}
            self.dataset[ids]['cost_vector'] = self.Vector_c(temp)
            PW = temp['windpower']
            b_sample = self.Vector_b(PW)
            self.dataset[ids]['b'] = b_sample
            self.dataset[ids]['z'], self.dataset[ids]['w'], self.dataset[ids]['constraints_dual'], _ = Oracle(
                self.dataset[ids]['cost_vector'], self, self.A, b=b_sample)
        print("dataset complete")
        return self.dataset

    def predict(self, x_test):
        """Predict the SPO prediction given x_test, when B is found."""
        if self.B is None:
            raise ValueError("B is not loaded. Please fit and load B first")
        else:
            return x_test @ self.B.T

########## DATA HANDLING ##########


def load_data(filename, config, split, parameters, small, poly_transformation=None, noise_level=None):
    """Loads and configures the dataset (features and targets), based on class/problem formulation"""
    # Read data and finding the number of samples (days)
    data = pd.read_pickle(filename)

    # Changing the name of the target columns for the two problems
    data = data.rename(columns={"Total_consumption_DK2": "P_D"})
    data = data.rename(columns={'OnshoreWindGe50kW_MWh_DK2': 'windpower'})

    if poly_transformation is None:
        data, _ = stat_features(config.Target, data, 3)

    Dates = data.timestamp.dt.date.unique().astype(str)
    n_samples = len(Dates)

    # ID column and hour column:
    data['ID'] = np.repeat(np.arange(n_samples), 24)
    data['hour'] = data['timestamp'].dt.hour

    # Using only the relevant hours:
    data = data[data['hour'] < config.LT]

    # Making a list of features
    non_features = config.Target + ["timestamp", "date", "ID", "hour"]
    features = list(set(data.columns)-set(non_features))

    numerical_features = data[features].select_dtypes(
        include=['number']).columns.tolist()
    if poly_transformation is None:
        features = variance_threshold_filter(data, numerical_features, 0.1)
    n_features = len(features)

    if poly_transformation is not None:
        data, features = add_poly(data, poly_transformation)

    # If we want a smaller dataset:
    if small == True:
        n_samples = 250
        data = data[data['ID'] < n_samples]
    else:
        data = data

    # Total number of (hourly) samples
    len_samples = config.LT*n_samples

    # Split into train and test ids
    IDS = data.ID.unique()
    train_test_split = int(split*len(IDS))
    train_ID = IDS[:train_test_split]
    test_ID = IDS[train_test_split:]
    n_samples_train = len(train_ID)
    n_samples_test = len(test_ID)
    print(f"Number of training samples: {n_samples_train}")
    print(f"Number of test samples: {n_samples_test}")

    if config.Problem == "EDP":
        """Problem configuration if EDP/OPF is considered"""
        data['windpower'] = data['windpower']/max(data['windpower'])
        data['P_W'] = np.array(data['windpower']) * config.P_W

        for n in config.N:
            data["P_D_N" + str(n)] = np.array(data.P_D) * \
                config.DemandFractions[n] - \
                (np.array(data['P_W']) if n == config.WN else 0)

        data = rescaler_v2(data, features, train_ID, test_ID)

        scaler = np.max(data[data['ID'].isin(IDS)]
                        [['P_D_N' + str(n) for n in config.N]])
        config.scaler = scaler
        config.base_scaler = scaler
        data.P_D = data.P_D/scaler
        data.P_W = data.P_W/scaler
        config.P_G = parameters['P_G']/scaler
        # config.sus = parameters['sus']/np.max(config.sus)
        config.P_F = parameters['P_F']/scaler
        for n in config.N:
            name = "P_D_N" + str(n)
            data[name] = data[name]/scaler
        non_features = non_features + \
            ['P_D_N' + str(n) for n in config.N] + ['P_W']
    else:
        """Problem configuration if producers is considered"""
        data['windpower'] = data['windpower']/max(data['windpower'])
        data['windpower'] = data['windpower']*parameters['PW_cap']
        DA_scaler = max(data['DA_DK2'])
        data['DA_DK2'] = data['DA_DK2']/DA_scaler
        scale_features = features
        data = rescaler_v2(data, scale_features, train_ID, test_ID)
        config.P_DIS_CH = parameters['P_DIS_CH']/parameters['PW_cap']
        config.SOC_init = parameters['SOC_init']/parameters['PW_cap']
        config.SOC_cap = parameters['SOC_cap']/parameters['PW_cap']
        config.PW_cap = parameters['PW_cap']/parameters['PW_cap']
        data['windpower'] = data['windpower']/parameters['PW_cap']

        print(f"PW {config.PW_cap}")
        print(f"P_DIS_CH {config.P_DIS_CH}")
        print(f"SOC_init {config.SOC_init}")
        print(f"SOC_cap {config.SOC_cap}")
        print(f"Efficiency {config.eff}")
        print(f"Max DA: {DA_scaler}")
        PW_scaler = parameters['PW_cap']
        scaler = PW_scaler*DA_scaler*7.45
        config.scaler = scaler
        config.base_scaler = PW_scaler
        config.DA_scaler = DA_scaler*7.45

    if noise_level is not None:
        print("Noise addition")
        data = add_noise(data, features, noise_factor=noise_level)
    if poly_transformation is None:
        data = temporal_features(data, 'timestamp')
        features = list(set(data.columns)-set(non_features))
    features.sort()
    # df = df.dropna()
    return data, IDS, train_ID, test_ID, features, scaler


def rescaler_v2(df, scale_features, train_ID, test_ID):
    """Performs min-max scaling on each feature    
    Parameter: 
    - DataFrame to scaling on
    - Features to scale on
    - Train ID to fit and transform on.
    - Test ID to transform on
    Returns: 
    - Scaled DataFrame
    """
    scaler_dict = {}

    # Split into train and test:
    df_train = df[df['ID'].isin(train_ID)]
    df_test = df[df['ID'].isin(test_ID)]

    # Fit and scale on training set:
    for feature in scale_features:
        if df_train[feature].dtype in [float, int]:
            scaler = MinMaxScaler()
            df_train[feature] = scaler.fit_transform(
                df_train[feature].values.reshape(-1, 1))
            df_test[feature] = scaler.transform(
                df_test[feature].values.reshape(-1, 1))

            scaler_dict[feature] = scaler
    df_scaled = pd.concat([df_train, df_test])
    return df_scaled


def stat_features(Target, df, stat_days):
    """
    Creates statistical features for the target variables.

    Parameters:
    - Target: List of target column names to calculate statistics for.
    - df: DataFrame containing the data.
    - stat_days: Number of days to use for the rolling statistics.

    Returns:
    - df: Original DataFrame with the new statistical features added.
    - stat_df: DataFrame containing only the statistical features.
    """
    stat_dfs = []
    quantiles = [0.25, 0.50, 0.75]
    statistics_names = ['Q25', 'Q50', 'Q75', 'mean']

    for targ in Target:
        df_temp = x_format_classifier(df, [targ], 24)

        # Prepare to store calculated statistics
        stat_tot = {f'{targ}_{q_name}': [] for q_name in statistics_names}

        # Loop over the data to calculate the rolling statistics
        for i in range(len(df_temp) - (stat_days - 1)):
            rolling_data = df_temp.iloc[i:i + stat_days, :].values.flatten()

            stat_tot[f'{targ}_Q25'].append(
                np.quantile(rolling_data, quantiles[0]))
            stat_tot[f'{targ}_Q50'].append(
                np.quantile(rolling_data, quantiles[1]))
            stat_tot[f'{targ}_Q75'].append(
                np.quantile(rolling_data, quantiles[2]))
            stat_tot[f'{targ}_mean'].append(np.mean(rolling_data))

        # Slice df_temp to align with the calculated statistics
        df_temp = df_temp.iloc[stat_days - 1:]

        # Add the calculated statistics to df_temp
        for stat_name in stat_tot:
            df_temp[stat_name] = stat_tot[stat_name]

        # Map statistical features back to the original DataFrame
        for stat_name in stat_tot:
            df[stat_name] = df['date'].map(df_temp[stat_name])

        # Append to stat_dfs for final concatenation
        stat_dfs.append(df_temp[list(stat_tot.keys())])

    # Combine all statistical features into a single DataFrame
    stat_df = pd.concat(stat_dfs, axis=1, join="outer")
    df = df.dropna()
    return df, stat_df


def temporal_features(df, date_column):
    """
    Adds temporal features to the DataFrame based on the date column, 
    encoding cyclical features using sine and cosine transformations.

    Parameters:
    - df: DataFrame containing the time series data.
    - date_column: Column name of the datetime data.

    Returns:
    - df: DataFrame with added temporal features, including sine and cosine 
           transformations for cyclical features.
    """
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Extracting standard temporal features
    # df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['hour_temp'] = df[date_column].dt.hour
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Sine and Cosine Transformations for Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_temp'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_temp'] / 24)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Adding Season Indicators
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'

    df['season'] = df['month'].apply(get_season)

    # One-hot encode the 'season' column
    df = pd.get_dummies(df, columns=['season'], prefix='', prefix_sep='')
    season_columns = ['Winter', 'Spring', 'Summer', 'Autumn']

    # Add any missing season columns with default value 0
    for col in season_columns:
        if col not in df.columns:
            df[col] = 0

    df[season_columns] = df[season_columns].astype(int)

    # Dropping the original cyclical columns
    df.drop(columns=['hour_temp', 'day_of_week',
            'day_of_year', 'month'], inplace=True)

    return df


def variance_threshold_filter(df, numerical_features, threshold):
    """
    Removes numerical features with a variance under the threshold.
    Performed column-wise.

    Parameters:
    - df: DataFrame
    - numerical_features: list of numerical features to check variance for
    - threshold: variance threshold level

    Returns:
    - Returns the list of numerical features that are not removed based on the variance threshold.
    """
    # Select only the numerical features from the DataFrame
    numerical_df = df[numerical_features]

    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numerical_df)

    variances = selector.variances_

    # Identify features with low variance
    low_variance_features = [
        feature for feature, variance in zip(numerical_features, variances)
        if variance < threshold
    ]

    # Print or log the low variance features
    # print(f"Features with variance below the threshold of {threshold}:")
    # for feature in low_variance_features:
    #     print(f"{feature}")
    # print()

    # Get the features that pass the variance threshold
    num_col = selector.get_support(indices=True)
    selected_features = [numerical_features[i] for i in num_col]

    return selected_features


def pairwise_correlation(df, features, correlation_threshold, plot=False):
    """
    Performs pairwise (pearson) correlation between features.
    Features are dropped based on variance.

    Parameters:
    -Dataframe with features.
    -List of features to calculate correlation on.
    -Correlation threshold - Cut of level of correlation.
    -Plot - If True, a correlation heatmap will be plotted before and after the pairwise correlation.

    Return:
    List of features.
    """
    df_corr = df[features].corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    # Pairs of high correlated features:
    high_corr_pairs = [(column, index)
                       for column in upper.columns for index in upper.index if upper.at[index, column] > correlation_threshold]
    if plot == True:
        plt.figure(figsize=(25, 25))
        sns.heatmap(df_corr, cmap='coolwarm', fmt='.2f')
        plt.show()

    # Use individual feature variation to determine which feature to drop:
    feature_variation = df[features].var()
    feature_remove = []
    for i in range(len(high_corr_pairs)):
        # print(f"Correlation: {df_corr[high_corr_pairs[i][0]][high_corr_pairs[i][1]]}")
        # print(f"{high_corr_pairs[i][0]}: {feature_variation[high_corr_pairs[i][0]]}")
        # print(f"{high_corr_pairs[i][1]}: {feature_variation[high_corr_pairs[i][1]]}")
        # print()
        if feature_variation[high_corr_pairs[i][0]] < feature_variation[high_corr_pairs[i][1]]:
            feature_remove.append(high_corr_pairs[i][0])
        else:
            feature_remove.append(high_corr_pairs[i][1])

    feature_remove = np.unique(feature_remove)
    features_filtered = list(set(features)-set(np.unique(feature_remove)))

    if plot == True:
        df_corr = df[features_filtered].corr().abs()
        plt.figure(figsize=(25, 25))
        sns.heatmap(df_corr, cmap='coolwarm', fmt='.2f')
        plt.show()

    return features_filtered


def pairwise_correlation_advanced(df, features, correlation_threshold, plot=False):
    """
    Performs pairwise (pearson) correlation between features.
    Features are dropped based on variance.
    Here the search for highly correlated features is extended to capture all instances.

    Parameters:
    -Dataframe with features.
    -List of features to calculate correlation on.
    -Correlation threshold - Cut of level of correlation.
    -Plot - If True, a correlation heatmap will be plotted before and after the pairwise correlation.

    Return:
    List of features.
    """
    df_corr = df[features].corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

    if plot:
        plt.figure(figsize=(25, 25))
        sns.heatmap(df_corr, cmap='coolwarm', fmt='.2f')
        plt.show()

    # Find groups of correlated features
    correlated_groups = defaultdict(set)
    for column in upper.columns:
        for index in upper.index:
            if upper.at[index, column] > correlation_threshold:
                correlated_groups[column].add(index)
                correlated_groups[index].add(column)

    def merge_groups(groups):
        merged_groups = []

        while groups:
            first_group = set(groups.pop(0))  # First group
            found_overlap = True

            while found_overlap:
                # set to false to control flow if no overlap is found.
                found_overlap = False
                remaining_groups = []

                for group in groups:
                    if first_group.intersection(group):
                        # Merge the intersecting groups
                        first_group.update(group)
                        # Control flow as another iteration is needed.
                        found_overlap = True
                    else:
                        # List of groups not intersecting.
                        remaining_groups.append(group)

                groups = remaining_groups

            # Add the fully merged group to the final list
            merged_groups.append(first_group)

        return merged_groups

    correlated_groups = merge_groups(list(correlated_groups.values()))

    # Select features to remove based on variance
    feature_variation = df[features].var()
    features_to_remove = []

    for group in correlated_groups:
        if len(group) > 1:
            # In each group, keep the feature with the highest variance
            sorted_group = sorted(
                group, key=lambda x: feature_variation[x], reverse=True)
            features_to_remove.extend(sorted_group[1:])

    features_filtered = list(set(features) - set(features_to_remove))

    # Plot the correlation matrix of filtered features if requested
    if plot:
        df_corr = df[features_filtered].corr().abs()
        plt.figure(figsize=(25, 25))
        sns.heatmap(df_corr, cmap='coolwarm', fmt='.2f')
        plt.show()

    return features_filtered


def mutual_info_simple(df, features, target, threshold_mean, threshold_target, plot=False):
    """
    Calculates mutual information between features and targets.
    Features are kept if: 
        - the mean mutual information is above the mean threshold these are kept.
        - the target-feature mutual information is above target threshold.

    Parameters:
    -Dataframe with features.
    -List of features to be used.
    -List of targets.
    -Information threshold mean - Cut of level for mean MI-score.
    -Information threshold target - Cut of level for target-feature MI-score.
    -Plot - If True, a bar plot of MI score against threshold level will be plotted.

    Return:
    -List of features.
    -Dataframe with MI scores.
    """
    mi_scores = [mutual_info_regression(
        df[features], df[tg], random_state=10) for tg in target]
    mi_results = pd.DataFrame(mi_scores, columns=features, index=target).T
    mi_results['mean'] = mi_results.mean(axis=1)

    if plot == True:
        mi_results.plot(kind='bar', figsize=(20, 10))
        plt.title(
            'Mutual Information Scores for Each Feature with Respect to Different Targets')
        plt.axhline(y=threshold_mean, color='red', linestyle='--',
                    label=f'Threshold = {threshold_mean}')
        plt.axhline(y=threshold_target, color='red', linestyle='--',
                    label=f'Threshold = {threshold_target}')
        plt.ylabel('MI Score')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc='best', ncol=3)
        plt.grid(alpha=0.4)
        # Show the plot
        plt.tight_layout()
        plt.show()

    mean_features = mi_results[mi_results['mean']
                               > threshold_mean].index.tolist()
    target_features_mi = mi_results[(
        mi_results > threshold_target).any(axis=1)].index.to_list()

    selected_features = list(np.unique(target_features_mi+mean_features))

    if len(selected_features) == 0:
        selected_features = [mi_results['mean'].idxmax()]
    return selected_features, mi_results


def mutual_information_advanced(df, features, target, threshold_mean, threshold_target, LT, plot=False):
    """
    Calculates mutual information between features and targets on daily basis.
    Features are kept if: 
        - the mean mutual information is above the mean threshold these are kept.
        - the target-feature mutual information is above target threshold.

    Parameters:
    -Dataframe with features.
    -List of features to be used.
    -List of targets.
    -Information threshold mean - Cut of level for mean MI-score.
    -Information threshold target - Cut of level for target-feature MI-score.
    -Plot - If True, a bar plot of MI score against threshold level will be plotted.

    Return:
    -List of features.
    -Dataframe with MI scores.
    """

    res_temp = np.zeros((len(features), len(target)))

    for idx, targ in enumerate(target):
        y_temp = np.array(df[targ]).reshape(-1, LT)
        for jdx, feat in enumerate(features):
            x_temp = np.array(df[feat]).reshape(-1, LT)
            mi_scores_h = np.array([mutual_info_regression(
                x_temp, y_temp[:, t]) for t in range(LT)])
            feature_score = np.mean(mi_scores_h, axis=0)
            res_temp[jdx, idx] = np.mean(feature_score)

    mi_results = pd.DataFrame(res_temp, columns=target, index=features)
    mi_results['mean'] = mi_results.mean(axis=1)

    if plot == True:
        mi_results.plot(kind='bar', figsize=(20, 10))
        plt.title(
            'Mutual Information Scores for Each Feature with Respect to Different Targets')
        plt.axhline(y=threshold_mean, color='red', linestyle='--',
                    label=f'Threshold = {threshold_mean}')
        plt.axhline(y=threshold_target, color='red', linestyle='--',
                    label=f'Threshold for Target = {threshold_target}')
        plt.ylabel('MI Score')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc='best', ncol=3)
        plt.grid(alpha=0.4)
        # Show the plot
        plt.tight_layout()
        plt.show()

    mean_features = mi_results[mi_results['mean']
                               > threshold_mean].index.tolist()
    target_features_mi = mi_results[(
        mi_results > threshold_target).any(axis=1)].index.to_list()

    selected_features = list(np.unique(target_features_mi+mean_features))

    return selected_features, mi_results


def feature_target_corr(df, features, Target, threshold_mean, threshold_target, plot=False):
    """
    Performs correlation (pearson) between features and targets.
    Features are dropped based on variance.

    Parameters:
    -Dataframe with features.
    -List of features to calculate correlation on.
    -Mean Correlation threshold - Cut of level of correlation on the mean correlation between targets and features.
    -Target Correlation threshold - Cut of level of correlation between targets and features.
    -Plot - If True, a bar plot of correlation level will be plotted.

    Return:
    List of features.
    Correlation matrix.
    """

    corr = pd.DataFrame(index=features, columns=Target)

    for feature in features:
        for target in Target:
            corr.loc[feature, target] = np.abs(
                np.corrcoef(df[target], df[feature])[0, 1])
    corr['mean'] = corr.mean(axis=1)

    mean_features = corr[corr['mean'] > threshold_mean].index.tolist()
    target_corr_features = corr[(
        corr > threshold_target).any(axis=1)].index.to_list()

    if plot == True:
        corr.plot(kind='bar', figsize=(20, 10))
        plt.title(
            'Pearson Correaltion for Each Feature with Respect to Different Targets')
        plt.axhline(y=threshold_target, color='red', linestyle='--',
                    label=f'Threshold for Target = {threshold_target}')
        plt.axhline(y=threshold_mean, color='red', linestyle='--',
                    label=f'Threshold for mean = {threshold_mean}')
        plt.ylabel('abs(Correlation)')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc='best', ncol=5)
        plt.grid(alpha=0.4)
        # Show the plot
        plt.tight_layout()
        plt.show()

    selected_features = list(np.unique(target_corr_features+mean_features))

    if len(selected_features) == 0:
        selected_features = [corr['mean'].idxmax()]

    return selected_features, corr


def shapley_values(config, target_model, data, train_ID, features, feature_length, threshold, plot=False, regularisation=None):
    """
    Calculates the shapley values for features of a given model.
    Hereafter features are chosen based on normalised information.

    Parameters:
    -Class of problem
    -List of features to calculate correlation on.
    -List with length of each feature - used for aggregation of SHAP values.
        Most important if daily stat features/intercept is used.
    -Threshold - Cut of level.
    -Plot - If True, a waterfall plot of aggregated SHAP values will be plotted.

    Return:
    List of features.
    Shapley values.
    """

    # First set-up small model to get predictions.
    data = data[:24*100]
    train_ID = train_ID[:100]
    # Here training data is split into train and validation.
    dataset_shapley = dataset_creator(config, data, train_ID, features)
    split = 0.8
    train_test_split_shap = int(split*(len(train_ID)))
    train_id_shap = train_ID[:train_test_split_shap]
    val_id_shap = train_ID[:train_test_split_shap]
    extract_train_data(config, train_id_shap)
    extract_test_data(config, val_id_shap)

    features.insert(0, "Intercept")
    config.features = features
    feature_length.insert(0, 1)

    if target_model == "SPO":
        if regularisation is None:
            B_shapley = SPO_ERM(config)
            model = config
        else:
            B_shapley = SPO_ERM(config, regularisation[0], regularisation[1])
            model = config
    elif target_model == "LR":
        model, _ = Linear_Regression(config)
    elif target_model == "DT":
        model, _ = Decision_Tree(config)

    background = shap.sample(config.x_train, 25)  # Sample 50 rows from X
    explainer = shap.KernelExplainer(model.predict, background, silent=True)
    shap_vals = explainer.shap_values(config.x_test, progress_message=None)

    # Transformation of shapley values of n_samples,n_features*n_hours,n_targets*n_hours to n_feautres:
    shap_val_abs = np.abs(shap_vals)
    # Sum across samples (axis=0)
    shap_val_general = shap_val_abs.sum(axis=0)
    # Sum across features by hour
    shap_val_feature_hourly = shap_val_general.sum(
        axis=1)  # Shape: (d*n_hours,)
    # Normalize
    shap_val_feature_hourly_norm = shap_val_feature_hourly / \
        shap_val_feature_hourly.sum()

    # Precompute indices for slicing - split into features
    # Adjusted to handle varying feature lengths
    indexes = np.cumsum([0] + feature_length)
    shap_features = [shap_val_feature_hourly_norm[indexes[i]:indexes[i+1]].sum() for i in range(len(features))]

    # indexes = np.cumsum([0]+feature_length)
    # shap_features = [shap_val_feature_hourly_norm[indexes[i]:indexes[i+1]].sum() for i in range(len(features))]
    shap_features = pd.DataFrame(
        shap_features, index=features, columns=["Shap value"])

    # Selecting the features which contributes to x% of importance.
    shap_features = shap_features.sort_values(by='Shap value', ascending=False)
    shap_features['Cum. Importance'] = shap_features['Shap value'].cumsum()
    selected_features = shap_features[shap_features['Cum. Importance'] <= threshold]

    # If No features are included, include the next feature
    if selected_features.empty:
        selected_features = shap_features.iloc[[0]]
    else:
        # If the cumulative importance of the last selected feature is below the threshold, include the next feature
        if selected_features['Cum. Importance'].iloc[-1] < threshold:
            next_feature = shap_features.iloc[len(selected_features)]
            selected_features = pd.concat(
                [selected_features, next_feature.to_frame().T])

    selected_feature_names = selected_features.index.tolist()

    if plot == True:
        # Waterfall plot
        plt.figure(figsize=(10, 6))
        initial = 0
        for i, (feature, value) in enumerate(zip(shap_features.index, shap_features['Shap value'])):
            plt.bar(i, value, bottom=initial, edgecolor='black')
            initial += value

        plt.axhline(y=threshold, color='r', linestyle='--',
                    label=f'Threshold ({threshold})')
        plt.xticks(range(len(shap_features)),
                   shap_features.index, rotation=45, ha='right')
        plt.ylabel('SHAP Value')
        plt.grid(alpha=0.4)
        plt.title('Waterfall Plot of SHAP Values by Feature')
        plt.show()

    return selected_feature_names, shap_features


def pca(data, train_ID, test_ID, features, threshold):
    """Transform the feature data by Principle Component Analysis with the specified variance explanation level"""
    non_features = data.columns.difference(features)

    traindata = data[data['ID'].isin(train_ID)][features]
    testdata = data[data['ID'].isin(test_ID)][features]

    pcafit = PCA(n_components=threshold)
    pcafit.fit(np.array(traindata))
    traindata_transformed = pcafit.transform(traindata)
    testdata_transformed = pcafit.transform(testdata)

    print(f"Number of input features: {traindata.shape[1]}")
    print(f"Number of components selected: {pcafit.n_components_}")
    print(
        f"Explained variance ratio of selected components: {pcafit.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pcafit.explained_variance_ratio_)}")

    new_features = [f"PC_{i}" for i in range(pcafit.n_components_)]
    traindata_transformed_df = pd.DataFrame(
        traindata_transformed, columns=new_features)
    testdata_transformed_df = pd.DataFrame(
        testdata_transformed, columns=new_features)
    data_transformed_df = pd.concat(
        (traindata_transformed_df, testdata_transformed_df))
    new_data = pd.concat([data_transformed_df.set_index(
        data[non_features].index), data[non_features]], axis=1)

    return new_data, new_features


def cross_validation(config, data, features, train_ID, k_fold, target_model, plot=False, Name=None, regularization=None):
    """
    Performs Cross-validation of a model based on the input data.
    The models are score on True SPO loss.
    The generalised error across is weighted for validation samples in each fold.

    Parameters:
    -Problem class
    -Dataframe with features.
    -List of features.
    -IDs for the training set.
    -K folds in the cross validation
    -Target model - The target model of the cross validation, i.e. the generalised error will be return for this model.
    -Plot - If True, the error for each fold for each model will be plotted.

    Return:
    Generalised error of the target model.
    Generalised error of all models.

    """

    dataset_cv = dataset_creator(
        config, data, train_ID, features)

    # Split train into train and validation data - k-fold timeseries
    tscv = TimeSeriesSplit(n_splits=k_fold)

    # Dict to save losses and list to save n_samples for each fold
    loss_dict = {}
    n_samples_fold = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_ID)):
        print()
        print(f"Fold {fold+1} out of {k_fold}")
        print()

        # Extract relevant train and val data from overall dataset:
        extract_train_data(config, train_idx)
        extract_test_data(config, val_idx)

        print("SPO ERM:")
        # From here all test is actually validation:
        if regularization is None:
            B_fold = SPO_ERM(config)
            spo_model_name = "SPO"
        else:
            B_fold = SPO_ERM(config, regularization[0], regularization[1])
            if regularization[0] == "Ridge":
                spo_model_name = f"SPO - Ridge - {regularization[1]}"
                target_model = spo_model_name
            elif regularization[0] == "Lasso":
                spo_model_name = f"SPO - Lasso - {regularization[1]}"
                target_model = spo_model_name
        print("c_SPO:")
        c_SPO_fold = config.predict(config.x_test)

        print("Comparison models:")
        # Comparison models:
        c_Lin_fold = Linear_Regression(config)[1]

        if regularization is None:
            c_Ridge_fold = RidgeRegression(config, alpha=1)[1]
            c_Lasso_fold = LassoRegression(config, alpha=1)[1]
        else:
            c_Ridge_fold = RidgeRegression(config, alpha=regularization[1])[1]
            c_Lasso_fold = LassoRegression(config, alpha=regularization[1])[1]

        c_RF_fold = RandForest(config)[1]
        c_DT_fold = Decision_Tree(config)[1]

        predictions = [c_SPO_fold, c_Lin_fold, c_Ridge_fold,
                       c_Lasso_fold, c_RF_fold, c_DT_fold]
        model_names = [spo_model_name, 'LR', 'Ridge', 'Lasso', 'RF', 'DT']

        # predictions = [c_SPO_fold]
        # model_names = [spo_model_name]  # , 'LR', 'Ridge', 'Lasso', 'RF', 'DT']

        w_preds, z_preds, duals, gen_loss_dict, loss_sample_dict, count_infeasibles = compare_results(
            predictions, model_names, config, False)

        print(
            f"No. infeasibles for SPO: {np.sum(count_infeasibles[spo_model_name])}")

        loss_dict[fold] = gen_loss_dict
        n_samples_fold.append(len(val_idx))

    # loss per model per fold:
    model_loss = {model: np.array([loss_dict[fold][model]['SPO loss']
                                  for fold in range(k_fold)]) for model in model_names}
    # Generalised model loss
    error_gen = {model: np.sum(
        (n_samples_fold / np.sum(n_samples_fold)) * model_loss[model]) for model in model_names}

    print()
    print(f"CV general loss for {target_model}: {error_gen[target_model]}")
    print()
    if plot == True:
        plt.figure()
        plt.title(f"Cross validation - Average loss for {Name}")
        for model in model_names:
            plt.plot(model_loss[model], label=model)
        plt.legend()
        plt.ylabel("True SPO loss [p.u]")
        plt.xlabel("Fold [-]")
        if Name is not None:
            plt.savefig(f'{Name}.png', dpi=300,
                        bbox_inches='tight', facecolor='white')
        plt.show()

    return error_gen[target_model], error_gen


def technique_selector(config, data, features, Target, target_model, train_ID, k_fold):
    """Greedy Algorithm for Feature selection technique selection. """
    print("Starting technique selection")
    # error_gen, _ = cross_validation(
    #     config, data, features, train_ID, k_fold, target_model, False)
    # best_performance = error_gen
    best_performance = np.inf
    # print(f"Baseline performance: {best_performance:.4f}")
    # best_technique = "Nothing"

    # print(f"Start technique: {best_technique} ")
    techniques_to_use = []
    counter = 0

    # techniques = ['Pairwise','Mutual_info','Feature_Target']
    techniques = ['Pairwise', 'Mutual_info', 'Feature_Target', 'Shapley']

    while len(techniques) > 0:
        print()
        print(f"Iteration {counter+1}")
        counter += 1
        print()

        loss_dict = {}
        for tech in techniques:
            if tech == 'Pairwise':
                print()
                print("Pairwise")
                print()
                reduced_features = pairwise_correlation(
                    data, features, 0.75, False)
                print("Chosen features:")
                print(reduced_features)
                print()
            elif tech == 'Mutual_info':
                print()
                print("MI")
                print()
                reduced_features, _ = mutual_info_simple(
                    data, features, Target, 0.4, 0.6, False)
                print("Chosen features:")
                print(reduced_features)
                print()
            elif tech == 'Feature_Target':
                print()
                print("Feature Target")
                print()
                reduced_features, _ = feature_target_corr(
                    data, features, Target, 0.65, 0.75, False)
                print("Chosen features:")
                print(reduced_features)
                print()
            elif tech == 'Shapley':
                print()
                print("Shapley")
                print()
                feature_length = [24 for i in range(len(features))]
                reduced_features, _ = shapley_values(
                    config, target_model, data, train_ID, features, feature_length, 0.6, False)
                print("Chosen features:")
                print(reduced_features)
                print()
            print()
            error_gen, _ = cross_validation(
                config, data, reduced_features, train_ID, k_fold, target_model, False)

            # Store both error and reduced features in loss_dict
            loss_dict[tech] = {'Error': error_gen,
                               'Features': reduced_features}
        print()

        best_technique = min(loss_dict, key=lambda x: loss_dict[x]['Error'])
        gen_loss_lowest = loss_dict[best_technique]['Error']

        if abs(gen_loss_lowest) < abs(best_performance):
            print()
            print(f"For iteration {counter}")
            best_performance = gen_loss_lowest
            print(f"Best performance: {best_performance:.4f}")
            print(f"Best technique: {best_technique}")
            print()

            # Use the features of the best technique
            reduced_features = loss_dict[best_technique]['Features']
            print(f"Reduced features: {reduced_features}")
            print()

            techniques_to_use.append(best_technique)
            features = reduced_features
            techniques.remove(best_technique)
            if len(features) == 1:
                print(f"Best techniques: {techniques_to_use}")
                print(f"Feature set: {features}")
                break
        else:
            print("No Improvement")
            print(f"Best techniques: {techniques_to_use}")
            print(f"Feature set: {features}")
            break
    return features, techniques_to_use


def PCA_tuning(config, thresholds, data, features, train_ID, test_ID, k_fold):
    """Tests PCA for different levels of variance explanaiton"""
    loss_dict = {}
    print()
    print("No PCA:")
    error_gen, model_loss = cross_validation(
        config, data, features, train_ID, k_fold, 'SPO', False)
    loss_dict['None'] = error_gen
    print("error_gen = ", error_gen)
    for threshold in thresholds:
        print()
        print("pca variance explained ratio: ", threshold)
        newdata, newfeatures = pca(
            data, train_ID, test_ID, features, threshold)
        error_gen, model_loss = cross_validation(
            config, newdata, newfeatures, train_ID, k_fold, 'SPO', False)
        loss_dict[threshold] = error_gen
        print("error_gen = ", error_gen)

    best_threshold = min(loss_dict, key=loss_dict.get)
    print("best threshold: ", best_threshold)

    if best_threshold == 'None':
        return data, features
    else:
        finaldata, finalfeatures = pca(
            data, train_ID, test_ID, features, best_threshold)
        return finaldata, finalfeatures


def Hyperparameter_tuning(config, data, features, train_ID, Regularisation_options, Regularisation_level, k_fold):
    """Tunes SPO (and Lasso and Ridge) models with regularisation.

    Parameteres:
    config: Class of problem.
    data: Dataset.
    feature: feature set.
    Train_ID: Train ID.
    Regularisation options: Lasso, Ridge.
    Regularisation level: Level of regularisation parameter alpha.

    Returns:
    Regularisation set-up for SPO.
    """
    spo_loss_dict = {}
    lasso_loss_dict = {}
    ridge_loss_dict = {}

    error_gen, model_loss = cross_validation(
        config, data, features, train_ID, k_fold, 'SPO', False)

    spo_loss_dict['None'] = error_gen
    if 'Ridge' in Regularisation_options:
        ridge_loss_dict['None'] = model_loss['Ridge']
    if 'Lasso' in Regularisation_options:
        lasso_loss_dict['None'] = model_loss['Lasso']

    for regularisation in Regularisation_options:
        for level in Regularisation_level:
            print(f"{regularisation} at {level}")
            error_gen, model_loss = cross_validation(
                config, data, features, train_ID, k_fold, 'SPO', False, [regularisation, level])
            spo_loss_dict[regularisation, level] = error_gen
            if regularisation == 'Ridge':
                ridge_loss_dict[level] = model_loss['Ridge']
            if regularisation == 'Lasso':
                lasso_loss_dict[level] = model_loss['Lasso']

    best_spo = min(spo_loss_dict, key=spo_loss_dict.get)
    min_spo_loss = spo_loss_dict[best_spo]

    print(f"Best SPO combination: {best_spo} - loss: {min_spo_loss:.5f}")

    if 'Ridge' in Regularisation_options:
        best_ridge = min(ridge_loss_dict, key=ridge_loss_dict.get)
        min_ridge_loss = ridge_loss_dict[best_ridge]
        print(
            f"Best LR Ridge level: {best_ridge} - loss: {min_ridge_loss:.5f}")

    if 'Lasso' in Regularisation_options:
        best_lasso = min(lasso_loss_dict, key=lasso_loss_dict.get)
        min_lasso_loss = lasso_loss_dict[best_lasso]
        print(
            f"Best LR Lasso level: {best_lasso} - loss: {min_lasso_loss:.5f}")

    spo_regularisation = [best_spo[0], best_spo[1]]

    return spo_regularisation


#################### DATA HANDLING FOR SPO ####################


def dataset_creator(config, data: pd.DataFrame, IDS: list, features: list) -> dict:
    """Creates the dataset for a given problem configuration
    Returns the dataset in a dictionary"""
    if config.Problem == 'Lagrangian':
        return config.dataset_creator(data, IDS, features)
    elif config.Problem == "Traders_Dual":
        return config.dataset_creator(data, IDS, features)
    elif config.Problem == "Equality":
        # print("Correct dataset creator")
        return config.dataset_creator(data, IDS, features)
    elif config.Problem == "Equality_one_uncertainty" or config.Problem == "Equality_one_uncertainty_seq":
        # print("Correct dataset creator")
        return config.dataset_creator(data, IDS, features)
    else:
        config.features = features
        config.dataset = {}
        for ids in IDS:
            temp = data[data['ID'] == ids]
            config.dataset[ids] = {col: temp[col].to_numpy()
                                   for col in temp.columns}
            config.dataset[ids]['cost_vector'] = config.Vector_c(temp)
            config.dataset[ids]['z'], config.dataset[ids]['w'], config.dataset[ids]['constraints_dual'], _ = Oracle(
                config.dataset[ids]['cost_vector'], config, config.A)
        return config.dataset


def spo_format(config, ID_label: list):
    """Converts the dictionary data into SPO format, i.e. vector/matrixes
    Returns c, x, w, dual and z"""
    if config.dataset is None:
        raise ValueError(
            "Dataset is not loaded. Please load the dataset first")
    c = np.array([config.dataset[ids]['cost_vector'] for ids in ID_label])
    x = np.array([np.array([config.dataset[ids][f] for f in config.features]).flatten()
                  for ids in ID_label])
    x = np.insert(x, 0, 1, axis=1)
    z = np.array([config.dataset[ids]['z'] for ids in ID_label])
    if config.Problem == 'Lagrangian':
        # w = np.array([np.concatenate([np.hstack([config.dataset[id][var]
        #                                          for var in config.variables[:-1]]), np.ones(config.LT)]) for id in ID_label])
        w = np.array([config.dataset[ids]['w'] for ids in ID_label])
        dual = np.zeros((w.shape))
    elif config.Problem == "Equality":
        w = np.array([config.dataset[ids]['w'] for ids in ID_label])
        dual = np.zeros((w.shape))
    else:
        w = np.array([config.dataset[ids]['w'] for ids in ID_label])
        dual = np.array([config.dataset[ids]['constraints_dual']
                         for ids in ID_label])
    if config.Problem == "Equality_one_uncertainty" or config.Problem == "Equality_one_uncertainty_seq":
        b = np.array([config.dataset[ids]['b'] for ids in ID_label])
        return c, x, w, dual, z, b
    return c, x, w, dual, z


def extract_train_data(config, train_ID: list):
    """Extracts the training data from the dictionary and saves it in the class instance"""
    print(f"Features used: {config.features}")
    if config.Problem == "Equality_one_uncertainty" or config.Problem == "Equality_one_uncertainty_seq":
        config.c_train, config.x_train, config.w_train, config.dual_train, config.z_train, config.b_train = spo_format(config,
                                                                                                                       train_ID)
    else:
        config.c_train, config.x_train, config.w_train, config.dual_train, config.z_train = spo_format(config,
                                                                                                       train_ID)
    config.n_samples_train = len(train_ID)


def extract_test_data(config, test_ID: list):
    """Extracts the test data from the dictionary and saves it in the class instance"""
    if config.Problem == "Equality_one_uncertainty" or config.Problem == "Equality_one_uncertainty_seq":
        config.c_test, config.x_test, config.w_test, config.dual_test, config.z_test, config.b_test = spo_format(config,
                                                                                                                 test_ID)
    else:
        config.c_test, config.x_test, config.w_test, config.dual_test, config.z_test = spo_format(config,
                                                                                                  test_ID)
    config.n_samples_test = len(test_ID)


def w_var_translator(w: np.ndarray, variable: list) -> pd.DataFrame:
    """Formats vector w into a dataframe, with variable names as columns.
    Returns a Dataframe"""
    res = pd.DataFrame(w.reshape(len(variable), -1).T, columns=variable)
    return res


#################### kmeans CLUSTERING ####################


def kmeans_clustering(df, cluster_target, n_clusters, LT):
    """Performs k-means clustering on the cluster target"""
    df_cluster = df[cluster_target]

    date_range = pd.to_datetime(df_cluster.index.date).unique()

    cluster_frames_list = [pd.DataFrame(np.array(
        df_cluster[cluster_targ]).reshape(-1, LT)) for cluster_targ in cluster_target]

    df_cluster = pd.concat(cluster_frames_list, axis=1)
    df_cluster = df_cluster.set_index(date_range)

    column_names = [
        [f'{cluster_targ}_{i+1}' for i in range(LT)] for cluster_targ in cluster_target]

    column_names_flatten = np.array(column_names).flatten()

    df_cluster.columns = column_names_flatten

    # kmeans = KMedoids(n_clusters=n_clusters, random_state=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster)

    df['Cluster'] = df.index.date
    df['Cluster'] = df['Cluster'].map(df_cluster['Cluster'])

    # Get the cluster centers (each center will have 24 * len(cluster_target) values)
    cluster_centers = kmeans.cluster_centers_
    medoids = kmeans.medoid_indices_  # This gives the indices of the medoids
    print()
    print(medoids)
    print()
    # Optionally, return the cluster centers along with the updated dataframe
    return df, cluster_centers


def tslearn_kmeans_clustering(df, cluster_targets, n_clusters, LT):
    """Performs k-means with DTW as the distance metric clustering on the cluster target"""
    # from tslearn.clustering import KMedoids
    df_cluster = df[cluster_targets]
    date_range = pd.to_datetime(df_cluster.index.date).unique()
    cluster_frames_list = [pd.DataFrame(np.array(df_cluster[cluster_targ]).reshape(-1, LT))
                           for cluster_targ in cluster_targets]

    df_cluster = pd.concat(cluster_frames_list, axis=1)
    df_cluster.index = date_range

    column_names = [
        [f'{cluster_targ}_{i+1}' for i in range(LT)] for cluster_targ in cluster_targets]
    column_names_flatten = np.array(column_names).flatten()
    df_cluster.columns = column_names_flatten

    # Prepare the data for tslearn (3D array: n_samples x n_timestamps x n_features)
    df_cluster_3d = df_cluster.values.reshape(
        len(date_range), LT, len(cluster_targets))

    # Apply KMeans clustering with DTW as the distance metric
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
                              metric="dtw", random_state=0)
    # kmeans = KMedoids(n_clusters=n_clusters,
    #                   metric="dtw", random_state=0)

    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster_3d)

    df['Cluster'] = df.index.date
    df['Cluster'] = df['Cluster'].map(df_cluster['Cluster'])

   # Get the cluster centers (each center is a time series)
    cluster_centers = kmeans.cluster_centers_

    return df, cluster_centers


def train_test_cluster_ID(df_cluster, train_ID, test_ID):
    """Assigns the cluster id to the train and test IDS"""
    n_clusters = len(df_cluster.Cluster.unique())

    train_ID_cluster = {
        cluster_idx: [
            ID for ID in train_ID if ID in df_cluster[df_cluster['Cluster'] == cluster_idx].ID.unique()]
        for cluster_idx in range(n_clusters)}

    test_ID_cluster = {
        cluster_idx: [
            ID for ID in test_ID if ID in df_cluster[df_cluster['Cluster'] == cluster_idx].ID.unique()]
        for cluster_idx in range(n_clusters)}

    return train_ID_cluster, test_ID_cluster

#################### MANUAL CLUSTERING ####################


def manual_clustering(df, price_levels, cluster_target, LT):
    """Performs manual clustering of a target based upon a statistical measure (i.e. quantiles)"""
    df_cluster = df[cluster_target]

    date_range = pd.to_datetime(df_cluster.index.date).unique()

    cluster_frames_list = [pd.DataFrame(np.array(
        df_cluster[cluster_targ]).reshape(-1, LT)) for cluster_targ in cluster_target]

    df_cluster = pd.concat(cluster_frames_list, axis=1)
    df_cluster = df_cluster.set_index(date_range)

    df_cluster['row_mean'] = df_cluster.mean(axis=1)

    price_levels.append(np.inf)
    price_levels.insert(0, -np.inf)
    labels = np.arange(0, len(price_levels)-1)
    df_cluster['Cluster'] = pd.cut(
        df_cluster['row_mean'], bins=price_levels, labels=labels)

    df['Cluster'] = df.index.date
    df['Cluster'] = df['Cluster'].map(df_cluster['Cluster'])

    cluster_centers = df_cluster.groupby(
        'Cluster').mean().drop(columns='row_mean')

    cluster_centers = np.array(cluster_centers)

    return df, cluster_centers


def x_format_classifier(df, features, LT):
    """Formats the feature array x, into SPO format"""
    X = df[features]
    date_range = pd.to_datetime(X.index.date).unique()

    X_frames_list = [pd.DataFrame(np.array(
        X[feature]).reshape(-1, LT)) for feature in features]
    X = pd.concat(X_frames_list, axis=1)

    X = X.set_index(date_range)
    X.columns = range(X.shape[1])
    return X


def y_format_classifier(df):
    """Formats the target into SPO format"""
    def get_classification(group):
        return group['Cluster'].mode()[0]

    grouped = df.groupby('date')

    classifications = grouped.apply(get_classification)

    df_classification_target = pd.DataFrame(classifications)
    df_classification_target.columns = ["Cluster"]
    return df_classification_target


def cluster_classification(df_train, df_test, features, LT):
    """Classifier to predict the classification of the test set"""
    # Formats data into dates. x is feature data, y is label
    x = x_format_classifier(df_train, features, LT)
    y = y_format_classifier(df_train)
    # Test data for predicition
    x_test = x_format_classifier(df_test, features, LT)
    y_test = y_format_classifier(df_test)

    # Splits the training data into test and train for the classifier
    split_int = int(x.shape[0]*0.8)
    X_train = x[:split_int]
    X_val = x[split_int:]
    Y_train = y[:split_int]
    Y_val = y[split_int:]

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_val.shape[0]}")

    # SVM model
    print()
    print("Training SVM classifier")
    # clf = SVC(kernel="linear")
    # clf = SVC()
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, Y_train)
    Y_pred_train = clf.predict(X_val)

    performance = accuracy_score(Y_val.values, Y_pred_train)
    print(f"Validation Accuracy is: {performance:.4f}")
    print("")

    # Predict cluster for test data set and map prediction to df_test:
    print("Predicting clusters")
    y_pred = clf.predict(x_test)

    performance = accuracy_score(y_test.values, y_pred)
    print(f"Test Accuracy is: {performance:.4f}")
    print("")
    # drop true cluster from df_test
    df_test = df_test.drop(columns='Cluster')

    x_test['Cluster'] = y_pred
    df_test['Cluster'] = df_test['date']
    df_test['Cluster'] = df_test['Cluster'].map(x_test['Cluster'])
    # Combine and return full dataset - now ready for regression
    df = pd.concat([df_train, df_test])
    return df


########## PROBLEMS ##########


def simple_EDP(G, cost_G, P_G, P_D, problem_type, printing):
    """Optimisation problem for a simple EDP set up"""
    data = pd.DataFrame()
    data["Cost"] = cost_G
    data["Max"] = P_G

    # Problem
    model = gp.Model("Simple EDP")
    model.Params.LogToConsole = 0
    if problem_type == "primal":
        p = model.addVars(G, vtype=GRB.CONTINUOUS, lb=0,
                          name="Generator production")
    elif problem_type == "dual":
        Lambda = model.addVar(vtype=GRB.CONTINUOUS, lb=-
                              GRB.INFINITY, name="lambda")
        mu = model.addVars(G, vtype=GRB.CONTINUOUS, lb=0, name="mu")

    # Objective function
    if problem_type == "primal":
        model.setObjective(
            (gp.quicksum(cost_G[g] * p[g] for g in G)), sense=GRB.MINIMIZE)
    elif problem_type == "dual":
        model.setObjective(
            (Lambda * P_D - gp.quicksum(mu[g] * P_G[g] for g in G)), sense=GRB.MAXIMIZE)

    # Constraints:
    if problem_type == "primal":
        balanceconstraint = model.addConstr(
            (gp.quicksum(p[g] for g in G) == P_D), name="Balance")
        capacityconstraints = model.addConstrs(
            (p[g] <= P_G[g] for g in G), name="Upper bound capacity")
    elif problem_type == "dual":
        pconstraints = model.addConstrs(
            (Lambda - mu[g] <= cost_G[g] for g in G), name="p constraint")

    # Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL:
        obj_simpleED = model.ObjVal
        p_res = np.zeros(len(G))
        mu_res = np.zeros(len(G))
        for g in G:
            if problem_type == "primal":
                p_res[g] = p[g].x
                mu_res[g] = capacityconstraints[g].Pi
            elif problem_type == "dual":
                p_res[g] = pconstraints[g].Pi
                mu_res[g] = mu[g].x

        if problem_type == "primal":
            lambda_res = balanceconstraint.Pi
        elif problem_type == "dual":
            lambda_res = Lambda.x

        data["Production"] = p_res
        data["mu"] = mu_res

        if printing == True:
            print("Optimal solution found:")
            print(f"Total cost: {obj_simpleED:.2f} DKK")
            print(f"Electricity price (lambda): {lambda_res:.2f} DKK")
            print()
            print("Generator data:")
            print(data)

    return data, obj_simpleED, lambda_res


def EDP_standard_form(config, PD):
    """Optimisation problem for a EDP/OPF"""
    G = config.G
    T = config.T
    N = config.N
    GN = config.GN
    NN = config.NN
    DN = config.DN
    sus = config.sus

    # Problem
    model_primal = gp.Model("EDP multihour nodal primal")
    model_primal.Params.LogToConsole = 0
    p_primal = model_primal.addVars(
        G, T, vtype=GRB.CONTINUOUS, lb=0, name="Generator production")
    theta_primal = model_primal.addVars(
        N, T, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="voltage angle")
    # Objective function
    model_primal.setObjective(gp.quicksum(
        config.cost_G[g] * p_primal[g, t] for g in G for t in T), sense=GRB.MINIMIZE)

    # Capacity constraints:
    balanceconstraints = model_primal.addConstrs((gp.quicksum(p_primal[g, t] for g in GN[n])
                                                  - gp.quicksum(sus[n, m] * (theta_primal[n, t]
                                                                             - theta_primal[m, t]) for m in NN[n])
                                                  >= gp.quicksum(PD[d, t] for d in DN[n]) for n in N for t in T), name="Balance")

    capacityconstraints = model_primal.addConstrs(
        (-p_primal[g, t] >= -config.P_G[g] for g in G for t in T), name="Upper bound capacity")

    transmissionconstraints = model_primal.addConstrs(
        (-sus[n, m] * (theta_primal[n, t] - theta_primal[m, t]) >= -config.P_F[n, m] for n in N for m in NN[n] for t in T), name="transmission capacity")
    thetazeroconstraints = model_primal.addConstrs(
        (theta_primal[0, t] == 0 for t in T), name="theta0")

    # Optimize
    model_primal.optimize()
    # Initialize all result variables with zero arrays
    p_res_primal = np.zeros((len(G), len(T)))
    theta_res_primal = np.zeros((len(N), len(T)))
    muG_res_primal = np.zeros((len(G), len(T)))
    muF_res_primal = np.zeros((len(N), len(N), len(T)))
    lambda_res_primal = np.zeros((len(N), len(T)))
    sigma1_res_primal = np.zeros(len(T))
    # Assumed dimensions from 'LN' to 'N' for consistency
    flow_primal = np.zeros((len(N), len(N), len(T)))

    if model_primal.status == GRB.OPTIMAL:
        for t in T:
            sigma1_res_primal[t] = thetazeroconstraints[t].Pi
            for n in N:
                lambda_res_primal[n, t] = balanceconstraints[n, t].Pi
                theta_res_primal[n, t] = theta_primal[n, t].x
                for m in NN[n]:
                    muF_res_primal[n, m,
                                   t] = transmissionconstraints[n, m, t].Pi
                    flow_primal[n, m, t] = sus[n, m] * \
                        (theta_res_primal[n, t] - theta_res_primal[m, t])
                    flow_primal[m, n, t] = sus[m, n] * \
                        (theta_res_primal[m, t] - theta_res_primal[n, t])
            for g in G:
                p_res_primal[g, t] = p_primal[g, t].x
                muG_res_primal[g, t] = capacityconstraints[g, t].Pi

        obj_multihourED_primal = model_primal.ObjVal
        # print("\nOptimal solution found:")
        # print(f"Total cost: {obj_multihourED_primal:.2f} DKK")
        infeasible = 0
    else:
        # If model is infeasible or suboptimal, we handle it here
        obj_multihourED_primal = 0.0
        infeasible = 1
    # Save results in a dictionary, regardless of feasibility
    sample_res = {
        'z': obj_multihourED_primal,
        'p': p_res_primal,
        'theta': theta_res_primal,
        'muG': muG_res_primal,
        'muF': muF_res_primal,
        'lambda': lambda_res_primal,
        'sigma1': sigma1_res_primal,
        'flow': flow_primal,
        'infeasibles': infeasible
    }
    return sample_res


def traders_prob_primary(DA_price, PW, config):
    """Optimisation problem for a producers/traders problem on inequality in power balance"""
    # Access needed info from config
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    eff = config.eff
    SOC_cap = config.SOC_cap
    SOC_init = config.SOC_init

    # Problem
    model = gp.Model("Trader WF BESS")
    model.Params.LogToConsole = 0
    p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="Wind production")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")
    # Objective function
    model.setObjective(
        (-gp.quicksum(DA_price[t] * (p_w[t] + p_dis[t]) for t in T)), sense=GRB.MINIMIZE)

    # #Capacity constraints:
    c1 = model.addConstrs((p_w[t] + p_ch[t] <= PW[t]
                          for t in T), name="Upper wind")
    c2 = model.addConstrs(
        (p_ch[t] <= P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (p_dis[t] <= P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                          1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((SOC[t] <= SOC_cap for t in T), name="Upper SOC")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        p_w_res = np.array([p_w[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        mu_res = np.array([c1[t].Pi for t in T])
        obj = model.ObjVal
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_Price': DA_price,
                      'PW': PW}
    else:
        sample_res = {'z': 0,  # To set the
                      'pW': np.zeros(config.LT),
                      'pCH': np.zeros(config.LT),
                      'pDIS': np.zeros(config.LT),
                      'SOC': np.zeros(config.LT),
                      'mu': np.zeros(config.LT),
                      'DA_Price': np.zeros(config.LT),
                      'PW': np.zeros(config.LT)}
        print("FUUUUCCKK")

    return sample_res


def producers_problem_standard_form_equality(config, DA, PW):
    """Optimisation problem for producers problem on with pw(pDA) = PW-pch, with equality for power balance"""
    # Traders SPO format min
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    # Problem
    model = gp.Model("Trader WF BESS")
    model.Params.LogToConsole = 0
    # p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="Wind production")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")
    # Objective function
    # model.setObjective(
    #     (-gp.quicksum(DA[t] * (p_w[t] + p_dis[t]) for t in T)), sense=GRB.MINIMIZE)

    # # #Capacity constraints:
    # c1 = model.addConstrs((-p_w[t] - p_ch[t] == -PW[t]
    #                        for t in T), name="Upper wind")
    model.setObjective(
        (-gp.quicksum(DA[t]*(PW[t]-p_ch[t]+p_dis[t]) for t in T)), sense=GRB.MINIMIZE)
    c1 = model.addConstrs((PW[t]-p_ch[t] >= 0 for t in T), name="pw")
    c2 = model.addConstrs(
        (-p_ch[t] >= -P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (-p_dis[t] >= -P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                           1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((-SOC[t] >= -SOC_cap for t in T), name="Upper SOC")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        p_w_res = np.array([PW[t]-p_ch[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        mu_res = np.array([c1[t].Pi for t in T])
        obj = model.ObjVal
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_DK2': DA,
                      'windpower': PW,
                      'infeasibles': 0}
        df_res = pd.DataFrame(sample_res)
    else:
        # print("Fuck")
        p_w_res = np.zeros(len(T))
        p_ch_res = np.zeros(len(T))
        p_dis_res = np.zeros(len(T))
        SOC_res = np.zeros(len(T))
        mu_res = np.zeros(len(T))
        obj = 0
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_DK2': DA,
                      'windpower': PW,
                      'infeasibles': 1}
        df_res = pd.DataFrame(sample_res)
    return sample_res


def producers_problem_standard_form(config, DA, PW):
    """Optimisation problem for producers problem with equality for power balance"""
    # Traders SPO format min
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    # Problem
    model = gp.Model("Trader WF BESS")
    model.Params.LogToConsole = 0
    p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="Wind production")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")
    # Objective function
    model.setObjective(
        (-gp.quicksum(DA[t] * (p_w[t] + p_dis[t]) for t in T)), sense=GRB.MINIMIZE)

    # #Capacity constraints:
    c1 = model.addConstrs((-p_w[t] - p_ch[t] == -PW[t]
                           for t in T), name="Upper wind")
    c2 = model.addConstrs(
        (-p_ch[t] >= -P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (-p_dis[t] >= -P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                           1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((-SOC[t] >= -SOC_cap for t in T), name="Upper SOC")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        p_w_res = np.array([p_w[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        mu_res = np.array([c1[t].Pi for t in T])
        obj = model.ObjVal
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_Price': DA,
                      'PW': PW,
                      'infeasibles': 0}
        df_res = pd.DataFrame(sample_res)
    else:
        # print("Fuck")
        p_w_res = np.zeros(len(T))
        p_ch_res = np.zeros(len(T))
        p_dis_res = np.zeros(len(T))
        SOC_res = np.zeros(len(T))
        mu_res = np.zeros(len(T))
        obj = 0
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_Price': DA,
                      'PW': PW,
                      'infeasibles': 1}
        df_res = pd.DataFrame(sample_res)
    return sample_res


def producers_problem_lagrangian_relaxation(config, DA, PW, mu):
    """Optimisation problem for producers problem on Lagrangian relaxation"""
    # Traders SPO format min
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    # Problem
    model = gp.Model("Trader WF BESS Lagrangian Relaxation")
    model.Params.LogToConsole = 0

    p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="Wind production")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")

    # Objective function with Lagrangian relaxation
    model.setObjective(
        (-gp.quicksum(DA[t] * (p_w[t] + p_dis[t]) for t in T)
         + gp.quicksum(mu[t] * (p_w[t] + p_ch[t] - PW[t]) for t in T)),
        sense=GRB.MINIMIZE)

    # Capacity constraints:
    # c1 = model.addConstrs((-p_w[t] >= -PW_nom for t in T), name = "PW Nom")
    c2 = model.addConstrs((-p_ch[t] >= -P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (-p_dis[t] >= -P_DIS_CH for t in T), name="Upper dis")
    c6 = model.addConstrs((-SOC[t] >= -SOC_cap for t in T), name="Upper SOC")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                          1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")

    # Solve the problem
    model.optimize()
    if model.status == GRB.OPTIMAL:
        p_w_res = np.array([p_w[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        obj = model.ObjVal
        sample_res = {'z': obj,
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'DA_Price': DA,
                      'PW': PW}
        # print(obj)
        # df_res = pd.DataFrame(sample_res)
    else:
        # print("Fuck")
        p_w_res = np.zeros(len(T))
        p_ch_res = np.zeros(len(T))
        p_dis_res = np.zeros(len(T))
        SOC_res = np.zeros(len(T))
        mu_res = np.zeros(len(T))
        obj = 0
        sample_res = {'z': obj,
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_Price': DA,
                      'PW': PW}
        # df_res = pd.DataFrame(sample_res)
    return sample_res


def producers_problem_relaxation(config, DA, PW):
    """Optimisation problem for producers problem with reformulation approach"""
    # Traders SPO format min
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    PW_nom = 1

    # Problem
    model = gp.Model("Trader WF BESS")
    model.Params.LogToConsole = 0
    p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="Wind production")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")
    # Objective function
    model.setObjective(
        (-gp.quicksum(DA[t] * (PW[t] - p_ch[t] + p_dis[t]) for t in T)), sense=GRB.MINIMIZE)

    # # #Capacity constraints:
    # c1 = model.addConstrs((-p_w[t] == -PW[t] + p_ch[t]
    #                         for t in T), name="Upper wind")
    # c1 = model.addConstrs((-p_w[t] >= -PW_nom for t in T), name="Upper wind")

    c2 = model.addConstrs(
        (-p_ch[t] >= -P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (-p_dis[t] >= -P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                           1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((-SOC[t] >= -SOC_cap for t in T), name="Upper SOC")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        p_w_res = np.array([p_w[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        # mu_res = np.array([c1[t].Pi for t in T])
        obj = model.ObjVal
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'DA_Price': DA,
                      'PW': PW,
                      'infeasibles': 0}
        df_res = pd.DataFrame(sample_res)
    else:
        # print("Fuck")
        p_w_res = np.zeros(len(T))
        p_ch_res = np.zeros(len(T))
        p_dis_res = np.zeros(len(T))
        SOC_res = np.zeros(len(T))
        mu_res = np.zeros(len(T))
        obj = 0
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'DA_Price': DA,
                      'PW': PW,
                      'infeasibles': 1}
    return sample_res


def producers_problem_one_uncertainty(config, DA, PW):
    """Optimisation problem for producers problem on where PW is known"""
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    # Problem
    model = gp.Model("Trader WF BESS")
    model.Params.LogToConsole = 0
    p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="Wind production")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")
    # Objective function
    model.setObjective(
        (-gp.quicksum(DA[t] * (p_w[t] + p_dis[t]) for t in T)), sense=GRB.MINIMIZE)

    # #Capacity constraints:
    c1 = model.addConstrs((-p_w[t] - p_ch[t] >= -PW[t]
                           for t in T), name="Upper wind")
    c2 = model.addConstrs(
        (-p_ch[t] >= -P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (-p_dis[t] >= -P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                           1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((-SOC[t] >= -SOC_cap for t in T), name="Upper SOC")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        p_w_res = np.array([p_w[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        mu_res = np.array([c1[t].Pi for t in T])
        obj = model.ObjVal
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_Price': DA,
                      'PW': PW}
        df_res = pd.DataFrame(sample_res)
    else:
        # print("Fuck")
        p_w_res = np.zeros(len(T))
        p_ch_res = np.zeros(len(T))
        p_dis_res = np.zeros(len(T))
        SOC_res = np.zeros(len(T))
        mu_res = np.zeros(len(T))
        obj = 0
        sample_res = {'z': obj,  # To set the
                      'pW': p_w_res,
                      'pCH': p_ch_res,
                      'pDIS': p_dis_res,
                      'SOC': SOC_res,
                      'mu': mu_res,
                      'DA_Price': DA,
                      'PW': PW}
        df_res = pd.DataFrame(sample_res)
    return sample_res


########## SPO ##########


def Oracle(c_pred: np.ndarray, config, A: np.ndarray, b=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Solves Any optimisation problen on:

    c^t.w  s.t.  A.w>=b

    Returns z, w, and duals of w"""
    Problem = config.Problem
    LT = config.LT
    w_len = config.w_len
    if (Problem != "Equality_one_uncertainty") and (Problem != "Equality_one_uncertainty_seq"):
        b = config.b
    model = gp.Model("Oracle of Delphi")
    model.Params.LogToConsole = 0

    w = model.addMVar(w_len, vtype=GRB.CONTINUOUS, name="Decision")
    model.setObjective(c_pred@w, sense=GRB.MINIMIZE)
    constraints = model.addConstr((A@w >= b))
    model.optimize()
    if model.status == GRB.OPTIMAL:
        z = model.ObjVal
        w = w.x
        duals = constraints.Pi
        count_infeasible = 0
    else:
        print("FUUUUUUCCKK")
        z = 0
        w = np.zeros(w_len)
        duals = np.zeros(b.shape[0])
        count_infeasible = 1
    return z, w, duals, count_infeasible


def SPO_ERM(config, regularization=None, regularization_val=None):
    """SPO ERM - Fits coefficient matrix B for SPO. Constraints are implemtend based on the class is performed for"""
    Problem = config.Problem
    print(f"Problem type: {Problem}")
    if Problem == "Equality_one_uncertainty":
        b = config.b_train
    elif Problem == "Equality_one_uncertainty_seq":
        b = config.b_train
    else:
        b = config.b
    A = config.A

    c = config.c_train
    x = config.x_train
    w = config.w_train
    z = config.z_train
    w_len = config.w_len
    n_samples_train = config.n_samples_train

    features_no = x.shape[1]
    LT = config.LT

    SPO_ERM = gp.Model("ERM of SPO+")
    SPO_ERM.Params.LogToConsole = 0

    B = SPO_ERM.addMVar((w_len, features_no), lb=-GRB.INFINITY)
    p = SPO_ERM.addMVar((n_samples_train, A.shape[0]),
                        vtype=GRB.CONTINUOUS, lb=0, name="Dual variable")

# Add auxiliary variables for Lasso regularization (absolute values of coefficients)
    if regularization == "Lasso":
        u = SPO_ERM.addMVar((w_len, features_no), lb=0,
                            name="Auxiliary variable for Lasso")

    print("SIZES:")
    print(f"A: {A.shape}")
    print(f"b: {b.shape}")
    print(f"c: {c.shape}")
    print(f"x: {x.shape}")
    print(f"w: {w.shape}")
    print(f"z: {z.shape}")
    print(f"B: {B.shape}")
    print(f"p: {p.shape}")
    print()

    if Problem == "Equality_one_uncertainty" or Problem == "Equality_one_uncertainty_seq":
        if regularization == "Ridge":
            # Use squared L2 norm for Ridge regularization
            SPO_ERM.setObjective(((1/n_samples_train) * gp.quicksum(
                -b[i].T @ p[i, :] + 2 * B @ x[i] @ w[i] - z[i] for i in range(n_samples_train)) +
                regularization_val * gp.quicksum(B[i, j] * B[i, j] for i in range(w_len) for j in range(features_no))),
                sense=GRB.MINIMIZE)

        elif regularization == "Lasso":
            SPO_ERM.setObjective(((1/n_samples_train) * gp.quicksum(
                -b[i].T @ p[i, :] + 2 * B @ x[i] @ w[i] - z[i] for i in range(n_samples_train)) +
                regularization_val * gp.quicksum(u[i, j] for i in range(w_len) for j in range(features_no))),
                sense=GRB.MINIMIZE)

        else:
            SPO_ERM.setObjective(
                ((1/n_samples_train)*gp.quicksum(-b[i].T@p[i, :]+2*B@x[i]@w[i]-z[i] for i in range(n_samples_train))), sense=GRB.MINIMIZE)
    else:
        if regularization == "Ridge":
            # Use squared L2 norm for Ridge regularization
            SPO_ERM.setObjective(((1/n_samples_train) * gp.quicksum(
                -b.T @ p[i, :] + 2 * B @ x[i] @ w[i] - z[i] for i in range(n_samples_train)) +
                regularization_val * gp.quicksum(B[i, j] * B[i, j] for i in range(w_len) for j in range(features_no))),
                sense=GRB.MINIMIZE)

        elif regularization == "Lasso":
            SPO_ERM.setObjective(((1/n_samples_train) * gp.quicksum(
                -b.T @ p[i, :] + 2 * B @ x[i] @ w[i] - z[i] for i in range(n_samples_train)) +
                regularization_val * gp.quicksum(u[i, j] for i in range(w_len) for j in range(features_no))),
                sense=GRB.MINIMIZE)

        else:
            SPO_ERM.setObjective(
                ((1/n_samples_train)*gp.quicksum(-b.T@p[i, :]+2*B@x[i]@w[i]-z[i] for i in range(n_samples_train))), sense=GRB.MINIMIZE)

    SPO_ERM.addConstrs((A.T@p[i, :] == 2*B@x[i]-c[i]
                        for i in range(n_samples_train)), name="Constraint")

    if config.Problem == "Equality":
        SPO_ERM.addConstrs(
            (B[-24:]@x[i] == 0 for i in range(n_samples_train)), name="c_soc=0")
        SPO_ERM.addConstrs((B[:24, :]@x[i] == -B[24:48, :]@x[i]
                           for i in range(n_samples_train)), name="lambda1=-lambda2")
        SPO_ERM.addConstrs((-B[2*24:3*24]@x[i]-(B[:24]@x[i]
                                                * w[i][:24]) >= 0 for i in range(n_samples_train)), name="pw>=0")

    if config.Problem == "EDP":
        SPO_ERM.addConstrs(
            (B[n*config.LT:(n+1)*config.LT]@x[i] <= 0 for n in config.N for i in range(n_samples_train)), name="PD")
        SPO_ERM.addConstrs(
            (B[config.LN*config.LT+t*config.LG:config.LN*config.LT+(t+1)*config.LG]@x[i] == config.P_G for t in config.T for i in range(n_samples_train)), name="PG")
        SPO_ERM.addConstrs(
            (B[config.LN*config.LT+config.LT*config.LG+l*config.LT:config.LN*config.LT+config.LT*config.LG+(l+1)*config.LT]@x[i] == config.P_F[config.TransmissionLines[l][0], config.TransmissionLines[l][1]] for l in config.L for i in range(n_samples_train)), name="PF")

    if regularization == "Lasso":
        # Add constraints for the auxiliary variables
        SPO_ERM.addConstrs((u[i, j] >= B[i, j] for i in range(w_len)
                            for j in range(features_no)), name="u >= B")
        SPO_ERM.addConstrs((u[i, j] >= -B[i, j] for i in range(w_len)
                            for j in range(features_no)), name="u >= -B")

    SPO_ERM.optimize()

    # Runtime warning can måske fjernes ved omformulering af matematikken.

    if SPO_ERM.status == GRB.OPTIMAL:
        print("Optimal solution found")
        # for d in range(w_len):
        #     B_res[d, :] = B[d, :].x
        B_res = np.array([B[i, :].x for i in range(w_len)])
    else:
        print("Optimal solution not found")
        B_res = np.zeros((w_len, (features_no)))

    config.B = B_res
    return B_res

########## OTHER REGRESSION MODELS ##########


def closed_form_beta(config):
    """Calculates beta values for linear regression by the arithmical expression.
    """
    beta = np.linalg.inv(
        config.x_train.T @ config.x_train) @ config.x_train.T @ config.c_train
    return beta


def prediction(config, beta):
    """Performs a prediction based on test data and a coefficent vector/matrix"""
    predictions = np.dot(config.x_test, beta)
    return predictions


def Linear_Regression(config) -> np.ndarray:
    """Uses a simple linear regressor to fit and predict c

    Returns Model and C_pred_lin"""
    lin_reg = LinearRegression()
    lin_reg.fit(config.x_train, config.c_train)
    c_pred_lin = lin_reg.predict(config.x_test)

    return lin_reg, c_pred_lin


def RidgeRegression(config, alpha) -> np.ndarray:
    """Uses a Ridge regressor to fit and predict c

    Returns Model and C_pred_Ridge"""
    ridge_reg = Ridge(alpha)
    ridge_reg.fit(config.x_train, config.c_train)
    c_pred_ridge = ridge_reg.predict(config.x_test)
    return ridge_reg, c_pred_ridge


def LassoRegression(config, alpha) -> np.ndarray:
    """Uses a Lasso regressor to fit and predict c

    Returns Model C_pred_Ridge"""
    lasso_reg = Lasso(alpha)
    lasso_reg.fit(config.x_train, config.c_train)
    c_pred_lasso = lasso_reg.predict(config.x_test)
    return lasso_reg, c_pred_lasso


def Decision_Tree(config) -> np.ndarray:
    """Uses a simple decision tree regressor to fit and predict c

    Returns Model C_pred_tree"""

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(config.x_train, config.c_train)
    c_pred_tree = tree_reg.predict(config.x_test)
    return tree_reg, c_pred_tree


def RandForest(config) -> np.ndarray:
    """Uses a simple Random Forest Regressor to fit and predict c.

    Returns C_pred_Forest"""
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(config.x_train, config.c_train)
    c_pred_forest = forest_reg.predict(config.x_test)
    return forest_reg, c_pred_forest

################################################## COMPARISON ##################################################


def SPO_loss(c_pred: np.ndarray, config, test_ID) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates True SPO loss for whole test set

    Returns: True SPO loss, z_pred and w_pred
    """

    if config.Problem == "Equality":
        losses, z_scale, w_preds_scale, dual_preds_scale, count_infeasible = SPO_loss_traders(
            c_pred, config, test_ID)
        config.w_test_scale = config.w_test*config.base_scaler
        config.z_test_scale = config.z_test*config.scaler
        return losses, z_scale, w_preds_scale, dual_preds_scale, np.array(count_infeasible)

    c_test = config.c_test
    z_test = config.z_test
    if config.Problem == "Equality_one_uncertainty":
        results = [Oracle(pred, config, config.A, config.b_test[i])
                   for i, pred in enumerate(c_pred)]
    elif config.Problem == "Equality_one_uncertainty_seq":
        results = [Oracle(pred, config, config.A, config.b_test_seq[i])
                   for i, pred in enumerate(c_pred)]
    else:
        results = [Oracle(pred, config, config.A) for pred in c_pred]
    w_preds = np.array([res[1] for res in results])
    z_preds = np.array([w_pred @ c_true for w_pred,
                       c_true in zip(w_preds, c_test)])
    dual_preds = np.array([res[2] for res in results])
    count_infeasible = np.array([res[3] for res in results])
    z_scale = z_preds * config.scaler
    config.z_test_scale = z_test * config.scaler
    losses = z_scale - config.z_test_scale

    if config.Problem == 'EDP':
        w_preds_scale = w_preds
        dual_preds = dual_preds  # * config.scaler
        config.dual_test_scale = config.dual_test  # *config.scaler
    else:
        w_preds = w_preds*config.base_scaler
        config.w_test_scale = config.w_test*config.base_scaler
        # config.z_test_scale = config.z_test * config.scaler

    return losses, z_scale, w_preds, dual_preds, count_infeasible


def SPO_loss_traders(prediction_spo, config, test_ID):
    """Special case for SPO loss for producers in the case of reformulation approach"""
    print("SPO loss traders")
    losses_spo = []
    z_real_spo = []
    w_pred_spo = []
    dual_pred_spo = []
    count_infeasibles_spo = []

    for i in range(len(prediction_spo)):
        # Clip and stick PW and DA from c_spo
        PW_spo = prediction_spo[i][2*24:3*24]/prediction_spo[i][24:2*24]
        DA_spo = prediction_spo[i][:24]

        # Solve the primary problem
        res = producers_problem_standard_form_equality(config, DA_spo, PW_spo)

        # Get decisions:
        pw_spo = res['pW']
        pch_spo = res['pCH']
        pdis_spo = res['pDIS']
        soc_spo = res['SOC']

        # get true DA and PW
        DA_true = config.dataset[test_ID[i]]['DA_DK2']
        # For PW to be relevant RT needs to be solved
        PW_true = config.dataset[test_ID[i]]['windpower']
        # Calculate realised DA profit:
        z_realised = -sum(DA_true*(pw_spo+pdis_spo))  # Only DA uncertainty
        # Both uncertainties
        # z_realised = -sum(DA_true*((PW_true-pch_spo)+pdis_spo))

        z_true = config.z_test[i]
        # calculate loss
        loss = (z_realised-z_true)*config.scaler
        # W skal være 1 her.
        w = np.array((pch_spo, pdis_spo, pw_spo, soc_spo)
                     ).flatten()*config.base_scaler
        # config.z_test_scale=z_true*config.scaler

        # save
        losses_spo.append(loss)
        z_real_spo.append(z_realised)
        w_pred_spo.append(w)
        dual_pred_spo.append(np.zeros(len(config.b)))
        count_infeasibles_spo.append(res['infeasibles'])

    return losses_spo, z_real_spo, w_pred_spo, dual_pred_spo, count_infeasibles_spo


def compare_results(predictions, model_names, config, test_ID, plot=False):
    """Evaluate models, plot losses and z_preds, and return w_preds. V1 - this should not be used"""

    # Dictionary to hold the predicted weights and z_preds
    w_preds_dict = {}
    z_preds_dict = {}
    dual_preds_dict = {}
    gen_loss_dict = {}
    loss_dict = {}
    count_infeasibles_dict = {}
    if plot == True:
        fig, axs = plt.subplots(2, 1, figsize=(12, 5))

    for i, (prediction, model_name) in enumerate(zip(predictions, model_names)):
        # Compute the SPO loss and metrics
        losses, z_preds, w_preds, dual_preds, count_infeasibles = SPO_loss(
            prediction, config, test_ID)

        mse_loss = [mean_squared_error(config.c_test[i], prediction[i])
                    for i in range(len(prediction))]

        mse_loss_gen = mean_squared_error(
            config.c_test.reshape(-1), prediction.reshape(-1))
        # Store w_preds and z_preds in the dictionaries
        w_preds_dict[model_name] = w_preds
        z_preds_dict[model_name] = z_preds
        dual_preds_dict[model_name] = dual_preds
        loss_dict[model_name] = {'SPO loss': losses, 'MSE': mse_loss}
        gen_loss_dict[model_name] = {
            'SPO loss': np.mean(losses), 'MSE': mse_loss_gen}
        count_infeasibles_dict[model_name] = count_infeasibles
        # Print metrics
        print(f"Model type: {model_name}")
        print(f"True SPO loss: {np.mean(losses):.3f}")
        print(f"MSE loss: {mse_loss_gen:.3f}")
        print()

        # Plot the loss and objective value
        if plot == True:
            axs[0].plot(losses, label=model_name)
            axs[1].plot(z_preds, label=model_name)

    if plot == True:
        # True SPO Loss plot
        axs[0].set_title("True SPO Loss Comparison")
        # axs[0].set_xlabel("Samples")
        axs[0].set_ylabel("Loss [-]")
        axs[0].grid(alpha=0.4)

        # Objective value plot
        axs[1].plot(config.z_test_scale, label="Target")
        axs[1].set_title("Predicted Objective value")
        axs[1].set_xlabel("Samples")
        axs[1].set_ylabel("Value [-] ")
        # plt.legend()
        axs[1].grid(alpha=0.4)

        plt.tight_layout()
        plt.legend(loc="upper center", bbox_to_anchor=(
            0.5, -0.3), fancybox=True, ncol=(len(model_names)+1))
        plt.show()

    return w_preds_dict, z_preds_dict, dual_preds_dict, gen_loss_dict, loss_dict, count_infeasibles_dict


def compare_results_v2(config, predictions_spo, predictions_sequential, test_ID, plot_res=False):
    """Evaluate models, plot losses and z_preds, and return w_preds. V2 - this should be used"""
    sample_loss = {}
    z_realised = {}
    w_prediction = {}
    dual_prediction = {}
    infeasibles = {}
    mse_loss = {}
    mse_seq = {}

    # Calculate SPO loss for vector models
    for model_name in predictions_spo.keys():
        losses, z_real, w_pred, dual_pred, count_infeasibles = SPO_loss(
            predictions_spo[model_name], config, test_ID)
        mse_loss[model_name] = [mean_squared_error(
            config.c_test[i], predictions_spo[model_name][i]) for i in range(len(test_ID))]
        sample_loss[model_name] = losses
        z_realised[model_name] = z_real
        w_prediction[model_name] = w_pred
        dual_prediction[model_name] = dual_pred
        infeasibles[model_name] = count_infeasibles
        print(f"Model type: {model_name}")
        print(f"Exp. SPO loss: {np.mean(sample_loss[model_name]):.4f}")
        print(f"MSE: {np.mean(mse_loss[model_name]):.4f}")
        print()

    predictions_spo['Target'] = config.c_test
    z_realised['Target'] = config.z_test_scale
    dual_prediction['Target'] = config.dual_test
    if config.Problem == 'EDP':
        w_prediction['Target'] = config.w_test
    else:
        w_prediction['Target'] = config.w_test_scale

    # Calculate SPO loss for sequential models
    for model_name in predictions_sequential.keys():
        losses_seq = []
        z_real_seq = []
        w_pred_seq = []
        dual_pred_seq = []
        count_infeasibles_seq = []

        mse_seq[model_name] = {targ: [mean_squared_error(
            config.dataset[test_ID[i]][targ], predictions_sequential[model_name][targ][i]) for i in range(len(test_ID))] for targ in config.Target}

        for idx in range(len(test_ID)):
            if config.Problem == "Equality":
                # DA and windpower predictions
                DA = predictions_sequential[model_name]['DA_DK2'][idx]
                PW = predictions_sequential[model_name]['windpower'][idx]
                # Compute DA schedule
                res = producers_problem_standard_form_equality(config, DA, PW)
                pw = res['pW']
                pch = res['pCH']
                pdis = res['pDIS']
                soc = res['SOC']

                # DA and windpower realisations
                DA_true = config.dataset[test_ID[idx]]['DA_DK2']
                PW_true = config.dataset[test_ID[idx]]['windpower']
                z_realised_val = -sum(DA_true*(pw+pdis))*config.scaler

                z_true = config.dataset[test_ID[idx]]['z']*config.scaler
                # SPO loss
                losses_seq.append(z_realised_val-z_true)
                # z_realised
                z_real_seq.append(z_realised_val)
                # decisions w
                w = np.array((pch, pdis, pw, soc)).flatten()*config.base_scaler
                w_pred_seq.append(w)
                # infeasibles
                count_infeasibles_seq.append(res['infeasibles'])
                # Duals are in not utilised in this problem - i.e.=0
                dual_pred_seq.append(np.zeros(len(config.b)))

            elif config.Problem == "EDP":
                # Get demand prediction
                PD_pred = np.array([predictions_sequential[model_name]
                                    [f'P_D_N{n}'][idx] for n in config.N])
                # Solve EDP/OPF
                res = EDP_standard_form(config, PD_pred)
                PD_true = np.array(
                    [config.dataset[test_ID[idx]][f'P_D_N{n}'] for n in config.N])
                # objective function of dual formulation
                z_realised_val = -sum(sum(res['lambda'][n, t] * PD_true[d, t] for n in config.N for d in config.DN[n]) - sum(res['muG'][g, t] * config.P_G[g]
                                                                                                                             for g in config.G) - sum(res['muF'][n, m, t] * config.P_F[n, m] for n in config.N for m in config.NN[n]) for t in config.T)*config.scaler
                z_true = config.dataset[test_ID[idx]]['z']*config.scaler
                losses_seq.append(z_realised_val-z_true)
                z_real_seq.append(z_realised_val)

                # special case of copperplate
                if len(config.N) == 1:
                    w = np.hstack((res['lambda'][0], res['muG'].T.flatten()))
                    dual = np.hstack(
                        (res['p'].T.flatten(), np.zeros(config.LT*5)))
                    count_infeasibles_seq.append(res['infeasibles'])
                    w_pred_seq.append(w)
                    dual_pred_seq.append(dual)
                else:
                    muFtest = np.zeros((config.LL, config.LT))
                    for l in config.TransmissionLines:
                        n = config.TransmissionLines[0]
                        m = config.TransmissionLines[1]
                        muFtest[l] = res['muF'][n, m]
                    w = np.hstack(
                        (res['lambda'].flatten(), res['muG'].T.flatten(), muFtest.flatten(), res['sigma1']))
                    dual = np.hstack(
                        (res['p'].T.flatten(), np.zeros(config.LN*config.LT), res['theta'].flatten(), np.zeros(config.LG*config.LT+config.LL*config.LT)))
                    count_infeasibles_seq.append(res['infeasibles'])
                    w_pred_seq.append(w)
                    dual_pred_seq.append(dual)

            elif config.Problem == "Equality_one_uncertainty":
                DA_pred = predictions_sequential[model_name]['DA_DK2'][idx]
                PW_true = config.dataset[test_ID[idx]]['windpower']
                res = producers_problem_standard_form_equality(
                    config, DA_pred, PW_true)
                pw = res['pW']
                pch = res['pCH']
                pdis = res['pDIS']
                soc = res['SOC']

                DA_true = config.dataset[test_ID[idx]]['DA_DK2']
                z_realised_val = - sum(DA_true*(pw+pdis))*config.scaler
                z_true = config.dataset[test_ID[idx]]['z']*config.scaler
                # SPO loss
                losses_seq.append(z_realised_val-z_true)
                # z_realised
                z_real_seq.append(z_realised_val)
                w = np.array((pch, pdis, pw, soc)).flatten()*config.base_scaler
                w_pred_seq.append(w)
                count_infeasibles_seq.append(res['infeasibles'])
                dual_pred_seq.append(np.zeros(config.A.shape[0]))

            elif config.Problem == "Equality_one_uncertainty_seq":
                DA_pred = predictions_sequential[model_name]['DA_DK2'][idx]
                PW_pred = predictions_sequential[model_name]['windpower'][idx]

                res = producers_problem_standard_form_equality(
                    config, DA_pred, PW_pred)
                pw = res['pW']
                pch = res['pCH']
                pdis = res['pDIS']
                soc = res['SOC']
                PW_true = config.dataset[test_ID[idx]]['windpower']
                DA_true = config.dataset[test_ID[idx]]['DA_DK2']
                z_realised_val = - sum(DA_true*(pw+pdis))*config.scaler
                z_true = config.dataset[test_ID[idx]]['z']*config.scaler
                # SPO loss
                losses_seq.append(z_realised_val-z_true)
                # z_realised
                z_real_seq.append(z_realised_val)
                w = np.array((pch, pdis, pw, soc)).flatten()*config.base_scaler
                w_pred_seq.append(w)
                count_infeasibles_seq.append(res['infeasibles'])
                dual_pred_seq.append(np.zeros(config.A.shape[0]))

            elif config.Problem == "Equality_original":
                DA = predictions_sequential[model_name]['DA_DK2'][idx]
                PW = predictions_sequential[model_name]['windpower'][idx]
                # Compute DA schedule
                res = producers_problem_relaxation(config, DA, PW)
                # DA and windpower realisations
                DA_true = config.dataset[test_ID[idx]]['DA_DK2']
                PW_true = config.dataset[test_ID[idx]]['windpower']
                # Objectiv value of DA schedule with realised DA and windpower
                z_realised_val = - \
                    np.sum(
                        DA_true * (PW_true - res['pCH'] + res['pDIS']))*config.scaler
                z_true = config.dataset[test_ID[idx]]['z']*config.scaler
                # SPO loss
                losses_seq.append(z_realised_val-z_true)
                # z_realised
                z_real_seq.append(z_realised_val)
                # decisions w
                w = np.hstack([res['pCH'], res['pDIS'],
                               np.ones(config.LT), res['SOC']])*config.base_scaler
                w_pred_seq.append(w)
                # infeasibles
                count_infeasibles_seq.append(res['infeasibles'])
                # Duals are in not utilised in this problem - i.e.=0
                dual_pred_seq.append(np.zeros(len(config.b)))

            elif config.Problem == "Lagrangian":
                DA = predictions_sequential[model_name]['DA_DK2'][idx]
                PW = predictions_sequential[model_name]['windpower'][idx]
                # Compute DA schedule
                res = producers_problem_standard_form_equality(config, DA, PW)
                # DA and windpower realisations
                DA_true = config.dataset[test_ID[idx]]['DA_DK2']
                PW_true = config.dataset[test_ID[idx]]['windpower']
                # Objectiv value of DA schedule with realised DA and windpower
                z_realised_val = - \
                    np.sum(
                        DA_true * (PW_true - res['pCH'] + res['pDIS']))*config.scaler
                z_true = config.dataset[test_ID[idx]]['z']*config.scaler
                # SPO loss
                losses_seq.append(z_realised_val-z_true)
                # z_realised
                z_real_seq.append(z_realised_val)
                # decisions w
                w = np.hstack([res['pCH'], res['pDIS'],
                               np.ones(config.LT), res['SOC']])*config.base_scaler
                w_pred_seq.append(w)
                # infeasibles
                count_infeasibles_seq.append(res['infeasibles'])
                # Duals are in not utilised in this problem - i.e.=0
                dual_pred_seq.append(np.zeros(len(config.b)))

        # Save solved samples for model
        sample_loss[model_name] = np.array(losses_seq)
        z_realised[model_name] = np.array(z_real_seq)
        w_prediction[model_name] = np.array(w_pred_seq)
        dual_prediction[model_name] = np.array(dual_pred_seq)
        infeasibles[model_name] = np.array(count_infeasibles_seq)

        print(f"Model type: {model_name}")
        print(f"Exp. SPO loss: {np.mean(sample_loss[model_name]):.4f}")
        for targ in config.Target:
            print(f"MSE for {targ}: {np.mean(mse_seq[model_name][targ]):.4f}")
        print()

    if plot_res == True:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for model_name in sample_loss.keys():
            ax.plot(sample_loss[model_name], label=f"{model_name}")
        plt.legend()
        ax.set_xlabel("Sample [n]")
        ax.set_ylabel("SPO loss [DKK]")
        ax.grid(alpha=0.4)
        ax.set_title("SPO loss")
        plt.show()

    gen_loss = {model: np.mean(sample_loss[model])
                for model in sample_loss.keys()}
    MSE = {'vector': mse_loss,
           'seq': mse_seq}
    return w_prediction, z_realised, dual_prediction, gen_loss, sample_loss, infeasibles, MSE


def spo_framework(config, data, IDS, train_ID, test_ID, plot_result, features, regularization=None):
    """Performs the SPO framework from A-Z with comparison regression models. - V1 should not be used"""
    dataset = dataset_creator(config, data, IDS, features)
    extract_train_data(config, train_ID)
    extract_test_data(config, test_ID)

    if regularization is None:
        B = SPO_ERM(config)
        spo_model_name = "SPO"
    else:
        B = SPO_ERM(config, regularization[0], regularization[1])
        if regularization[0] == "Ridge":
            spo_model_name = f"SPO - Ridge - {regularization[1]}"
        elif regularization[0] == "Lasso":
            spo_model_name = f"SPO - Lasso - {regularization[1]}"

    # SPO
    c_SPO = config.x_test@B.T

    _, c_LR = Linear_Regression(config)
    _, c_Ridge = RidgeRegression(
        config, alpha=1)
    _, c_Lasso = LassoRegression(
        config, alpha=1)
    _, c_RF = RandForest(config)
    _, c_DT = Decision_Tree(config)

    # Results:
    # Predictions and model names
    predictions = [c_SPO, c_LR, c_Ridge, c_Lasso, c_DT, c_RF]

    model_names = [spo_model_name, 'LR', 'Ridge', 'Lasso', 'DT', 'RF']

    if config.Problem == 'EDP':
        c_SPO_mod = np.zeros((c_SPO.shape))
        for s in range(len(test_ID)):
            c_SPO_mod[s, :config.LN*config.LT] = c_SPO[s, :config.LN*config.LT]
            c_SPO_mod[s, config.LN*config.LT:] = config.c_test[s,
                                                               config.LN*config.LT:]
        predictions = [c_SPO, c_SPO_mod, c_LR, c_Ridge, c_Lasso, c_DT, c_RF]
        model_names = [spo_model_name, spo_model_name +
                       ' - mod', 'LR', 'Ridge', 'Lasso', 'DT', 'RF']

    c_predictions = {model_names[i]: predictions[i]
                     for i in range(len(predictions))}

    # Call the function to evaluate, plot, and get w_preds and z_preds
    w_preds, z_preds, dual_preds, gen_loss_dict, loss_dict, count_infeasibles = compare_results(
        predictions, model_names, config, test_ID, plot_result)

    c_predictions['Target'] = config.c_test
    z_preds['Target'] = config.z_test_scale
    if config.Problem == 'EDP':
        w_preds['Target'] = config.w_test
        dual_preds['Target'] = config.dual_test_scale
    else:
        w_preds['Target'] = config.w_test_scale
        dual_preds['Target'] = config.dual_test

    if plot_result == True:
        if config.Problem == 'EDP':
            # predictions.append(config.c_test)
            # w_preds['Target'] = config.w_test
            # dual_preds['Target'] = config.dual_test
            model_names = model_names + ['Target']

            results = results_EDP(
                config, c_predictions, model_names, w_preds, dual_preds, True, True, 0)

            plot_sample = 0
            # # Time axis plots: ------------------------------------------------
            if config.LT > 1:
                # Demand predictions for each node:
                plot_EDP_nodal(model_names, np.arange(
                    config.LN), results["P_D"], plot_sample, "P_D for node ", config.LT, "Hours", 0)
                # Maximal generator production predictions for each generator:
                plot_EDP_nodal(model_names, np.arange(
                    config.LG), results["P_G"], plot_sample, "P_G for generator ", config.LT, "Hours", 1)
                # Maximal flow predictions through each line:
                if config.LL > 0:
                    plot_EDP_nodal(model_names, np.arange(
                        config.LL), results["P_F"], plot_sample, "P_F for line ", config.LT, "Hours", 0)
                # DA-price for each node:
                plot_EDP_nodal(model_names, np.arange(
                    config.LN), results["DA_price"], plot_sample, "DA_price for node ", config.LT, "Hours", 0)
                # Generator production for each node:
                plot_EDP_nodal(model_names, np.arange(
                    config.LG), results["generator_production"], plot_sample, "Production for generator ", config.LT, "Hours", 1)
                # Power flow through each line:
                if config.LL > 0:
                    plot_EDP_nodal(model_names, np.arange(
                        config.LL), results["flows"], plot_sample, "Flow through line ", config.LT, "Hours", 0)

        else:
            # w_preds['Target'] = config.w_test
            model_names = model_names + ['Target']

            plot_sample = 1
            c_spo = c_SPO[plot_sample]
            c_lr = c_LR[plot_sample]
            c_dt = c_DT[plot_sample]
            c_ridge = c_Ridge[plot_sample]
            c_lasso = c_Lasso[plot_sample]
            c_rf = c_RF[plot_sample]
            c_test = config.c_test[plot_sample]

            c_plot = [c_spo, c_lr, c_ridge, c_lasso, c_dt, c_rf, c_test]

            if config.Problem == 'Equality':
                plot_subplots_equality2(c_plot, model_names, config.LT)
            elif config.Problem == 'Lagrangian':
                plot_subplots_lagrangian2(
                    c_plot, model_names, config.LT, figsize=(20, 10))
            decision_plot(w_preds, config.variables, model_names, plot_sample)

    return gen_loss_dict, w_preds, c_predictions, dual_preds, loss_dict, z_preds, B, count_infeasibles


def spo_framework_v2(config, data, IDS, train_ID, test_ID, features, plot_res=False, regularisation=None):
    """Performs the SPO framework from A-Z with comparison regression models. - V2 should be used"""
    # Function for running SPO A-Z with comparisons to other regression models.
    dataset = dataset_creator(config, data, IDS, features)
    extract_train_data(config, train_ID)
    extract_test_data(config, test_ID)
    print("data extracted")

    if regularisation is None:
        B = SPO_ERM(config)
        spo_model_name = "SPO"
        regularisation = [0, 1, 1]
    else:
        B = SPO_ERM(config, regularisation[0], regularisation[1])
        if regularisation[0] == "Ridge":
            spo_model_name = f"SPO - Ridge - {regularisation[1]}"
        elif regularisation[0] == "Lasso":
            spo_model_name = f"SPO - Lasso - {regularisation[1]}"
    print("SPO fit")
    # SPO
    c_SPO = config.x_test@B.T

    # Classic regression model on the cost vector.
    c_LR = Linear_Regression(config)[1]
    c_Ridge = RidgeRegression(config, alpha=regularisation[1])[1]
    c_Lasso = LassoRegression(config, alpha=regularisation[2])[1]
    c_RF = RandForest(config)[1]
    c_DT = Decision_Tree(config)[1]

    if config.Problem == "Equality" or "Equality_one_uncertainty":
        predictions_spo = {spo_model_name: c_SPO}
    else:
        predictions_spo = {spo_model_name: c_SPO,
                           'LR_vector': c_LR,
                           'Ridge_vector': c_Ridge,
                           'Lasso_vector': c_Lasso,
                           'RF_vector': c_RF,
                           'DT_vector': c_DT}

    if config.Problem == 'EDP':
        c_SPO_mod = np.zeros((c_SPO.shape))
        for s in range(len(test_ID)):
            c_SPO_mod[s, :config.LN*config.LT] = c_SPO[s, :config.LN*config.LT]
            c_SPO_mod[s, config.LN*config.LT:] = config.c_test[s,
                                                               config.LN*config.LT:]
        spo_model_name2 = spo_model_name+" - mod"
        predictions_spo[spo_model_name2] = c_SPO_mod

    regularisation_level = regularisation[1:]

    # Sequential models:
    predictions_sequential, mse_sequential = sequential_models_v2(
        config, config.Target, data, train_ID, test_ID, features, regularisation_level)

    w_prediction, z_realised, dual_prediction, gen_loss, sample_loss, infeasibles, MSE = compare_results_v2(
        config, predictions_spo, predictions_sequential, test_ID, plot_res=False)

    # predictions_spo['Target'] = config.c_test
    # z_realised['Target'] = config.z_test_scale
    # dual_prediction['Target'] = config.dual_test
    # if config.Problem == 'EDP':
    #     w_prediction['Target'] = config.w_test
    # else:
    #     w_prediction['Target'] = config.w_test_scale

    return gen_loss, w_prediction, predictions_spo, predictions_sequential, dual_prediction, sample_loss, z_realised, B, infeasibles


def results_EDP(config, c_preds, model_names, w_preds, dual_preds, plotc, plotw, plot_sample):
    """Sorts and plots results for EPD v1 -should not be used"""
    n_samples_test = config.n_samples_test
    LN = config.LN
    LT = config.LT
    LG = config.LG
    LL = config.LL
    N = config.N
    T = config.T
    L = config.L

    PD_preds = {}
    PG_preds = {}
    PF_preds = {}
    DA_prices = {}
    muG_vars = {}
    muF_vars = {}
    generator_productions = {}
    theta_vars = {}
    flows = {}
    for i, model_name in enumerate(model_names):
        PD_pred = np.zeros((n_samples_test, LN, LT))
        DA_price = np.zeros((n_samples_test, LN, LT))
        PG_pred = np.zeros((n_samples_test, LT, LG))
        PF_pred = np.zeros((n_samples_test, LL, LT))
        muG_var = np.zeros((n_samples_test, LT, LG))
        muF_var = np.zeros((n_samples_test, LL, LT))
        generator_production = np.zeros((n_samples_test, LT, LG))
        theta_var = np.zeros((n_samples_test, LN, LT))
        flow = np.zeros((n_samples_test, LL, LT))
        for sample in range(n_samples_test):
            for n in N:
                PD_pred[sample, n, :] = - \
                    c_preds[model_name][sample][n*LT:(n+1)*LT]
                DA_price[sample, n,
                         :] = w_preds[model_name][sample][n*LT:(n+1)*LT]
                theta_var[sample, n, :] -= dual_preds[model_name][sample][LT *
                                                                          LG+n*LT:LT*LG+(n+1)*LT]
                theta_var[sample, n, :] += dual_preds[model_name][sample][LT *
                                                                          LG+LN*LT+n*LT:LT*LG+LN*LT+(n+1)*LT]
            for t in T:
                PG_pred[sample, t, :] = c_preds[model_name][sample][LN *
                                                                    LT+t*LG:LN*LT+(t+1)*LG]
                muG_var[sample, t, :] = w_preds[model_name][sample][LN *
                                                                    LT+t*LG:LN*LT+(t+1)*LG]
                generator_production[sample, t,
                                     :] = dual_preds[model_name][sample][t*LG:(t+1)*LG]
            for l in L:
                muF_var[sample, l, :] = w_preds[model_name][sample][LN *
                                                                    LT+LT*LG+l*LT:LN*LT+LT*LG+(l+1)*LT]
                PF_pred[sample, l, :] = c_preds[model_name][sample][LN *
                                                                    LT+LT*LG+l*LT:LN*LT+LT*LG+(l+1)*LT]
                n = config.TransmissionLines[l][0]
                m = config.TransmissionLines[l][1]
                for t in T:
                    flow[sample, l, t] = config.sus[n, m] * \
                        (theta_var[sample, n, t] - theta_var[sample, m, t])
        PD_preds[model_name] = PD_pred
        PG_preds[model_name] = PG_pred
        PF_preds[model_name] = PF_pred
        DA_prices[model_name] = DA_price
        muG_vars[model_name] = muG_var
        muF_vars[model_name] = muF_var
        generator_productions[model_name] = generator_production
        theta_vars[model_name] = theta_var
        flows[model_name] = flow

    results = {"P_D": PD_preds,
               "P_G": PG_preds,
               "P_F": PF_preds,
               "DA_price": DA_prices,
               "mu_G": muG_vars,
               "mu_F": muF_vars,
               "generator_production": generator_productions,
               "theta": theta_vars,
               "flows": flows}

    if plotc == True:
        plt.figure(figsize=(12, 6))
        for i, model_name in enumerate(model_names):
            plt.plot(c_preds[model_name][plot_sample], label=model_name)
        plt.title("C-vector for sample " + str(plot_sample))
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.legend(loc="lower center", bbox_to_anchor=(
            0.5, -0.15), fancybox=True, ncol=len(model_names))
        plt.show()

    if plotw == True:
        plt.figure(figsize=(12, 6))
        for i, model_name in enumerate(model_names):
            plt.step(np.arange(config.w_len),
                     w_preds[model_name][plot_sample], label=model_name)
        plt.title("w-vector for sample " + str(plot_sample))
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.legend(loc="lower center", bbox_to_anchor=(
            0.5, -0.15), fancybox=True, ncol=len(model_names))
        plt.show()

    return results


def results_EDP_v2(config, test_ID, model_names, w_prediction, dual_prediction, predictions_spo, predictions_sequential):
    """Sorts and plots results for EPD v2 -should be used"""
    LN = config.LN
    LT = config.LT
    LG = config.LG
    LL = config.LL
    N = config.N
    T = config.T
    L = config.L
    n_samples = len(test_ID)

    PDtest_dict = {}
    PGtest_dict = {}
    PFtest_dict = {}
    lambdatest_dict = {}
    muGtest_dict = {}
    muFtest_dict = {}
    pgtest_dict = {}
    thetatest_dict = {}
    flowtest_dict = {}

    modeltypes = ['SPO' for i in range(
        len(predictions_spo))] + ['seq' for i in range(len(predictions_sequential))]

    for i, model in enumerate(model_names):
        PDtest = np.zeros((n_samples, LN, LT))
        PGtest = np.zeros((n_samples, LT, LG))
        PFtest = np.zeros((n_samples, LL, LT))
        lambdatest = np.zeros((n_samples, LN, LT))
        muGtest = np.zeros((n_samples, LT, LG))
        muFtest = np.zeros((n_samples, LL, LT))
        pgtest = np.zeros((n_samples, LT, LG))
        thetatest = np.zeros((n_samples, LN, LT))
        flowtest = np.zeros((n_samples, LL, LT))
        for s in range(n_samples):
            for n in N:
                if modeltypes[i] == 'SPO':
                    PDtest[s, n] = - predictions_spo[model][s][n*LT:(n+1)*LT]
                elif modeltypes[i] == 'seq':
                    PDtest[s,
                           n] = predictions_sequential[model][f"P_D_N{n}"][s]
                lambdatest[s, n] = w_prediction[model][s][n*LT:(n+1)*LT]
                thetatest[s, n] -= dual_prediction[model][s][LT *
                                                             LG+n*LT:LT*LG+(n+1)*LT]
                thetatest[s, n] += dual_prediction[model][s][LT *
                                                             LG+LN*LT+n*LT:LT*LG+LN*LT+(n+1)*LT]
            for t in T:
                if modeltypes[i] == 'SPO':
                    PGtest[s, t] = predictions_spo[model][s][LN *
                                                             LT+t*LG:LN*LT+(t+1)*LG]
                elif modeltypes[i] == 'seq':
                    PGtest[s, t] = config.P_G
                muGtest[s, t] = w_prediction[model][s][LN *
                                                       LT+t*LG:LN*LT+(t+1)*LG]
                pgtest[s, t] = dual_prediction[model][s][t*LG:(t+1)*LG]
            for l in L:
                if modeltypes[i] == 'SPO':
                    PFtest[s, l] = predictions_spo[model][s][LN *
                                                             LT+LT*LG+l*LT:LN*LT+LT*LG+(l+1)*LT]
                elif modeltypes[i] == 'seq':
                    PFtest[s, l] = [config.P_F[config.TransmissionLines[l]
                                               [0], config.TransmissionLines[l][1]] for t in T]
                muFtest[s, l] = w_prediction[model][s][LN *
                                                       LT+LT*LG+l*LT:LN*LT+LT*LG+(l+1)*LT]
                n = config.TransmissionLines[l][0]
                m = config.TransmissionLines[l][1]
                for t in T:
                    flowtest[s, l, t] = config.sus[n, m] * \
                        (thetatest[s, n, t] - thetatest[s, m, t])

        PDtest_dict[model] = PDtest
        PGtest_dict[model] = PGtest
        PFtest_dict[model] = PFtest
        lambdatest_dict[model] = lambdatest
        muGtest_dict[model] = muGtest
        muFtest_dict[model] = muFtest
        pgtest_dict[model] = pgtest
        thetatest_dict[model] = thetatest
        flowtest_dict[model] = flowtest

    return PDtest_dict, PGtest_dict, PFtest_dict, lambdatest_dict, muGtest_dict, muFtest_dict, pgtest_dict, thetatest_dict, flowtest_dict

###################### MODEL EVALUATION/EXPERIMENTS #####################


def train_test_experiment(config, data, IDS, train_ID_sets, test_ID_set, gap, features, regu=None, plot=False, save_name=None):
    """Experiment to determine model reliance on train/test split"""
    test_ID = IDS[-test_ID_set:]

    traintest_gen_losses = {}
    traintest_losses = {}
    traintest_count_infeasibles = {}

    for train_size in train_ID_sets:
        print("Train size: ", train_size)
        train_ID = IDS[-(train_size + test_ID_set + gap):-(test_ID_set + gap)]

        gen_loss, w_prediction, predictions_spo, predictions_sequential, dual_prediction, losses, z_realised, B, count_infeasibles = spo_framework_v2(config, data, IDS, train_ID,
                                                                                                                                                      test_ID, features, False, regu)
        traintest_gen_losses[train_size] = gen_loss
        traintest_losses[train_size] = losses
        traintest_count_infeasibles[train_size] = count_infeasibles

    if plot:
        # Plot different sets of train and test data for targets
        # Create a color gradient from light green to dark green
        num_colors = len(train_ID_sets)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "green_gradient", ["lightgreen", "darkgreen"], N=num_colors)

        for target in config.Target:
            plt.figure(figsize=(10, 5))
            plt.title(f"Train and Test Sizes - for {target}", fontsize=14)

            temp = data[[target, 'ID']]  # Selecting target and ID columns

            # Track the data already plotted to avoid overlapping
            already_plotted_ids = []

            # Loop over each training set size
            for i in range(len(train_ID_sets)):
                # Get the train IDs excluding those already plotted
                train_ID = IDS[-(train_ID_sets[i] +
                                 test_ID_set + gap):-(test_ID_set + gap)]
                # Remove already plotted IDs
                train_ID = np.setdiff1d(train_ID, already_plotted_ids)
                # Add new IDs to the tracking list
                already_plotted_ids.extend(train_ID)
                # Get the gradient color for the current training set size
                color = cmap(i / num_colors)
                # Plot only new training data for the current set with gradient color
                plt.plot(temp[temp['ID'].isin(train_ID)][target],
                         label=f"{train_ID_sets[i]} train samples", linewidth=2, color=color, zorder=-i)

            # Plot test data with a distinct color and style
            plt.plot(temp[temp['ID'].isin(test_ID)][target],
                     label=f"{test_ID_set} Test samples", color='red', lw=2, alpha=0.8)

            # Plot the gap as a grey line
            gap_range = IDS[-(gap+test_ID_set):-test_ID_set]
            plt.plot(temp[temp['ID'].isin(gap_range)][target],
                     color='lightgrey', alpha=0.8, linewidth=2, label='Gap')

            # Customize the plot
            plt.grid(alpha=0.4)
            plt.legend(loc='upper left', fontsize=12)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel(f'{target} [P.U]', fontsize=12)
            if save_name is not None:
                plt.savefig(f'{save_name}_{target}.png', dpi=300,
                            bbox_inches='tight', facecolor='white')
            plt.show()

    return traintest_gen_losses, traintest_losses, traintest_count_infeasibles


def polynomial_degree_experiment(config, features_baseline, parameters, poly_degrees, plot=False, regularization=None):
    """Experiment to determine effect of polynomial features"""
    poly_gen_losses = {}
    poly_losses = {}
    poly_count_infeasibles = {}

    if plot == True:
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    for poly_deg in poly_degrees:
        poly_transformation = {'poly_deg': poly_deg,
                               'features': features_baseline}
        data, IDS, train_ID, test_ID, features_poly, scaler = load_data(
            'DataV4_shifted.pkl', config, 0.8, parameters, True, poly_transformation=poly_transformation, noise_level=None)

        gen_loss, w_prediction, predictions_spo, predictions_sequential, dual_prediction, sample_loss, z_realised, B, infeasibles = spo_framework_v2(config, data, IDS, train_ID,
                                                                                                                                                     test_ID, features_poly, False, regularization)
        # spo_gen_losses = {model: results['SPO loss']
        #                   for model, results in gen_loss.items()}
        poly_gen_losses[poly_deg] = gen_loss
        results = pd.DataFrame(poly_gen_losses).T
        # spo_losses = {model: results['SPO loss']
        #               for model, results in losses.items()}
        poly_losses[poly_deg] = sample_loss
        poly_count_infeasibles[poly_deg] = infeasibles

        if plot == True:
            ax[0].plot(data['DA_DK2_Lag'][:24*3], label=f"{poly_deg}")
            ax[1].plot(data['OnshoreWindGe50kW_MWh_DK2_Lag']
                       [:24*3], label=f"{poly_deg}")
            ax[2].plot(data['mean_wind_speed_DK2'][:24*3], label=f"{poly_deg}")

    if plot == True:
        ax[0].set_title("DA_DK2_Lag with polynomial transformation")
        ax[0].set_ylabel("Price [p.u]")
        ax[0].grid(alpha=0.4)
        ax[0].legend()

        ax[1].set_title("Windpower_Lag with polynomial transformation")
        ax[1].set_ylabel("Production [p.u]")
        ax[1].grid(alpha=0.4)
        ax[1].legend()
        ax[1].set_xlabel("Hours [h]")

        ax[2].set_title("mean_wind_speed_DK2 polynomial transformation")
        ax[2].set_ylabel("Production [p.u]")
        ax[2].grid(alpha=0.4)
        ax[2].legend()
        ax[2].set_xlabel("Hours [h]")
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 7))
        num_models = len(results.columns)
        bar_width = 0.1  # Width of the bars

        # Use a categorical index for x-axis
        training_size_categories = range(len(results.index))
        training_size_labels = [f'{size}' for size in results.index]

        for idx, model in enumerate(results.columns):
            # Offset bars for each model
            positions = np.array(training_size_categories) + idx * bar_width
            ax.bar(positions, results[model].to_numpy(),
                   width=bar_width, label=model, alpha=0.8)

        # Customize the plot
        ax.set_title(
            'SPO Loss Evolution with Increasing Nonlinear feature data', fontsize=14)
        ax.set_xlabel('Polynomial Degree', fontsize=12)
        ax.set_ylabel('SPO Loss', fontsize=12)
        ax.set_xticks(np.array(training_size_categories) +
                      bar_width * (num_models - 1) / 2)
        ax.set_xticklabels(training_size_labels, rotation=45, ha='right')
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.show()

    return poly_gen_losses, poly_losses, poly_count_infeasibles


def polynomial_degree_experiment_v2(config, data, IDS, train_ID, test_ID, features, non_features, poly_degrees, plot=False, regularisation=None):
    """Experiment to determine effect of polynomial features"""
    poly_gen_losses = {}
    poly_data_dict = {}
    poly_losses = {}
    poly_count_infeasibles = {}

    for poly_deg in poly_degrees:

        poly = PolynomialFeatures(poly_deg, include_bias=False)
        PCs_poly = poly.fit_transform(data[features])
        poly_features = ['poly_' + str(i) for i in range(PCs_poly.shape[1])]

        poly_data = data[non_features]
        poly_data[poly_features] = PCs_poly

        gen_losses, w_predictions, c_predictions, dual_predictions, losses, z_predictions, B, count_infeasibles = spo_framework_v2(
            config, poly_data, IDS, train_ID, test_ID, False, poly_features, plot_res=False, regularisation=regularisation)
        spo_gen_losses = {model: results['SPO loss']
                          for model, results in gen_losses.items()}
        spo_losses = {model: results['SPO loss']
                      for model, results in losses.items()}
        poly_gen_losses[poly_deg] = spo_gen_losses
        poly_losses[poly_deg] = spo_losses
        poly_count_infeasibles[poly_deg] = count_infeasibles
        poly_data_dict[poly_deg] = poly_data
        results = pd.DataFrame(poly_gen_losses).T

    if plot == True:
        fig, ax = plt.subplots(figsize=(12, 7))
        num_models = len(results.columns)
        bar_width = 0.1  # Width of the bars

        # Use a categorical index for x-axis
        training_size_categories = range(len(results.index))
        training_size_labels = [f'{size}' for size in results.index]

        for idx, model in enumerate(results.columns):
            # Offset bars for each model
            positions = np.array(training_size_categories) + idx * bar_width
            ax.bar(positions, results[model].to_numpy(),
                   width=bar_width, label=model, alpha=0.8)

        # Customize the plot
        ax.set_title(
            'SPO Loss Evolution with Increasing Nonlinear feature data', fontsize=14)
        ax.set_xlabel('Polynomial Degree', fontsize=12)
        ax.set_ylabel('SPO Loss', fontsize=12)
        ax.set_xticks(np.array(training_size_categories) +
                      bar_width * (num_models - 1) / 2)
        ax.set_xticklabels(training_size_labels, rotation=45, ha='right')
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.show()

    return poly_gen_losses, poly_losses, poly_count_infeasibles, poly_data_dict


def add_noise(data, features, noise_factor=0.1):
    """Add random Gaussian noise to the dataset.

    Args:
        data (pd.DataFrame): The original dataset.
        noise_factor (float): A scaling factor for the noise (e.g., 0.1 for 10% noise).

    Returns:
        pd.DataFrame: A new dataset with added noise.

    """

    np.random.seed(10)
    noisy_data = data.copy()
    for feature in features:
        #         # Add Gaussian noise proportional to the standard deviation of each feature
        noise = noise_factor * \
            np.random.randn(len(data[feature])) * data[feature].std()
        noisy_data[feature] += noise
    return noisy_data


def add_poly(data, poly_transformation):
    data_feature = data[poly_transformation['features']]
    poly = PolynomialFeatures(
        degree=poly_transformation['poly_deg'], include_bias=False)
    poly_arr = poly.fit_transform(data_feature)
    poly_feature_names = poly.get_feature_names_out(
        poly_transformation['features'])
    poly_data = pd.DataFrame(
        poly_arr, columns=poly_feature_names, index=data.index)
    data_non_feature = data.drop(columns=poly_transformation['features'])
    data = pd.concat([data_non_feature, poly_data], axis=1)
    features = poly_feature_names

    return data, features


def noise_experiment(config, data, IDS, train_ID, test_ID, features, Target, noise_levels, noise_target, noise_data, plot, regularisation):
    """Experiment to determine effect of noise in model"""
    if noise_target == 'Features':
        noise_target = features
    elif noise_target == 'Target':
        noise_target = Target
    elif noise_target == 'Both':
        noise_target = features+Target

    train = data[data['ID'].isin(train_ID)]
    test = data[data['ID'].isin(test_ID)]

    noise_gen_losses = {}
    noise_losses = {}
    noise_count_infeasibles = {}

    if plot == True:
        fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    for noise_level in noise_levels:
        print("Add noise = ", noise_level)
        # Controlling noise on train/test/both
        if noise_data == 'train':
            print("TRAIN")
            train_noisy = add_noise(
                train, noise_target, noise_factor=noise_level)
            data_noise = pd.concat([train_noisy, test])
        elif noise_data == 'test':
            print("TEST")
            test_noisy = add_noise(
                test, noise_target, noise_factor=noise_level)
            data_noise = pd.concat([train, test_noisy])
        elif noise_data == 'Both':
            print("BOTH")
            data_noise = add_noise(
                data, noise_target, noise_factor=noise_level)

        gen_loss, w_prediction, predictions_spo, predictions_sequential, dual_prediction, sample_loss, z_realised, B, infeasibles = spo_framework_v2(config, data_noise, IDS, train_ID,
                                                                                                                                                     test_ID, features, False, regularisation)

        noise_gen_losses[noise_level] = gen_loss
        noise_losses[noise_level] = sample_loss
        noise_count_infeasibles[noise_level] = infeasibles

        ax[0].plot(data_noise[features[0]][:24*7], label=f"{noise_level}")
        ax[1].plot(data_noise[features[1]][:24*7],
                   label=f"{noise_level}")
        ax[2].plot(data_noise[features[2]]
                   [:24*7], label=f"{noise_level}")

    if plot == True:
        ax[0].set_title(features[0])
        ax[0].set_ylabel("[-]")
        ax[0].grid(alpha=0.4)
        ax[0].legend()

        ax[1].set_title(features[1])
        ax[1].set_ylabel("[-]")
        ax[1].grid(alpha=0.4)
        ax[1].legend()
        ax[1].set_xlabel("Hours [h]")

        ax[2].set_title(features[2])
        ax[2].set_ylabel("[-]")
        ax[2].grid(alpha=0.4)
        ax[2].legend()
        ax[2].set_xlabel("Hours [h]")
        plt.show()

    return noise_gen_losses, noise_losses, noise_count_infeasibles


def Clustering_experiment(config, data, IDS, train_ID, test_ID, reduced_features, regularisation, cluster_target, method, n_clusters, manual_levels, plot=False, save_name=None):
    """Experiment to determine effect of clustering on target"""
    clustering_dict = {}
    # print("No clustering")

    gen_loss, w_prediction, predictions_spo, predictions_sequential, dual_prediction, sample_loss, z_realised, B, infeasibles = spo_framework_v2(config, data, IDS, train_ID, test_ID,
                                                                                                                                                 reduced_features, False, regularisation)
    clustering_dict['None'] = {'gen_loss': gen_loss,
                               'w_prediction': w_prediction,
                               'predictions_spo': predictions_spo,
                               'predictions_sequential': predictions_sequential,
                               'dual_prediction': dual_prediction,
                               'sample_loss': sample_loss,
                               'z_realised': z_realised,
                               'B': B,
                               'infeasibles': infeasibles}
    LT = config.LT
    # Split data into train and test set:
    # For training clustering classifier
    df_train = data[data['ID'].isin(train_ID)]
    # Real test for regression models:
    df_test = data[data['ID'].isin(test_ID)]

    if method == "Kmeans_TS":
        df_train, center_train = tslearn_kmeans_clustering(
            df_train, cluster_target, n_clusters, LT)
        df_test, center_test = tslearn_kmeans_clustering(
            df_test, cluster_target, n_clusters, LT)
    elif method == "Kmeans_SK":
        df_train, center_train = kmeans_clustering(
            df_train, cluster_target, n_clusters, LT)
        df_test, center_test = kmeans_clustering(
            df_test, cluster_target, n_clusters, LT)
    elif method == "Manual":
        df_train, center_train = manual_clustering(
            df_train, manual_levels.copy(), cluster_target, LT)
        df_test, center_test = manual_clustering(
            df_test, manual_levels.copy(), cluster_target, LT)

    # Train classifier and predict on test set.
    df_cluster = cluster_classification(
        df_train, df_test, reduced_features, LT)
    train_ID_cluster, test_ID_cluster = train_test_cluster_ID(
        df_cluster, train_ID, test_ID)

    n_clusters = df_train.Cluster.nunique()
    print(f"Number of cluster determined {n_clusters}")

    for i in range(n_clusters):
        print(f"Cluster {i+1}")
        print(f"Training samples: {len(train_ID_cluster[i])}")
        print(f"Testing samples: {len(test_ID_cluster[i])}")
        print()

    if plot == True:
        if len(cluster_target) == 1:
            plot_cluster(df_train, n_clusters,
                         cluster_target, center_train, LT, save_name)
        else:
            plot_cluster_multitarget(
                df_train, n_clusters, cluster_targets=cluster_target, cluster_center=center_train, LT=LT, save_name=save_name)

    non_empty_clusters = []

    for cluster_idx in range(n_clusters):
        if len(train_ID_cluster[cluster_idx]) > 0 and len(test_ID_cluster[cluster_idx]) > 0:
            non_empty_clusters.append(cluster_idx)
        else:
            del train_ID_cluster[cluster_idx]
            del test_ID_cluster[cluster_idx]

    for cluster_idx in non_empty_clusters:

        gen_loss, w_prediction, predictions_spo, predictions_sequential, dual_prediction, sample_loss, z_realised, B, infeasibles = spo_framework_v2(
            config, data, IDS, train_ID_cluster[cluster_idx], test_ID_cluster[cluster_idx], reduced_features, False, regularisation)

        clustering_dict[f'Cluster {cluster_idx+1}'] = {'gen_loss': gen_loss,
                                                       'w_prediction': w_prediction,
                                                       'predictions_spo': predictions_spo,
                                                       'predictions_sequential': predictions_sequential,
                                                       'dual_prediction': dual_prediction,
                                                       'sample_loss': sample_loss,
                                                       'z_realised': z_realised,
                                                       'B': B,
                                                       'infeasibles': infeasibles}

    return clustering_dict
############################################ COMPARISON MODELS ############################################


def sequential_models(config, data, train_ID, test_ID, features, Target, regu_level):
    """Fits and predicts the conventional regression models - v1, should not be used"""
    model_classes = {
        'LR': LinearRegression(),
        'Ridge': Ridge(regu_level),
        'Lasso': Lasso(regu_level),
        'DT': DecisionTreeRegressor(random_state=42),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Pre-split data into train and test once
    data_train = data[data['ID'].isin(train_ID)]
    data_test = data[data['ID'].isin(test_ID)]

    # Format the training and test data once for all targets
    x_train = np.array(x_format_classifier(data_train, features, config.LT))
    x_test = np.array(x_format_classifier(data_test, features, config.LT))

    # Initialize containers for predictions and results
    predictions_sequential = {}

    # Train and predict for each target
    for targ in Target:
        y_train = np.array(x_format_classifier(data_train, [targ], config.LT))
        y_test = np.array(x_format_classifier(data_test, [targ], config.LT))

        predictions_sequential[targ] = {}

        # Loop over all models and train/predict
        for model_name, model in model_classes.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)

            # Store predictions and MSE
            predictions_sequential[targ][model_name] = {
                'mse': mse,
                'predictions': y_pred

            }

    # List of model names for evaluation
    model_names = list(model_classes.keys())

    # Initialize evaluation containers
    loss_sequential = {}
    z_sequential = {}
    w_sequential = {}
    gen_loss_sequential = {}
    infeasibles_sequential = {}
    # Evaluate each model
    for model in model_names:
        z_realised = []
        loss_sample = []
        w = []
        infeasibles = []
        for idx in range(len(test_ID)):
            DA = predictions_sequential['DA_DK2'][model]['predictions'][idx]
            PW = predictions_sequential['windpower'][model]['predictions'][idx]

            # Compute relaxation result
            res = producers_problem_relaxation(config, DA, PW)

            # Stack and store results
            w.append(np.hstack([res['pCH'], res['pDIS'],
                     np.ones(config.LT), res['SOC']]))
            infeasibles.append(res['infeasibles'])
            # Calculate z_realised and loss for this sample
            DA_true = config.dataset[test_ID[idx]]['DA_DK2']
            PW_true = config.dataset[test_ID[idx]]['windpower']
            z_realised_val = - \
                np.sum(
                    DA_true * (PW_true - res['pCH'] + res['pDIS']))*config.scaler
            z_realised.append(z_realised_val)
            loss_sample.append(
                z_realised_val - config.dataset[test_ID[idx]]['z']*config.scaler)

        # Store results for this model
        loss_sequential[model] = {'SPO loss': loss_sample}
        z_sequential[model] = z_realised
        w_sequential[model] = w
        gen_loss_sequential[model] = np.mean(loss_sample)
        infeasibles_sequential[model] = np.array(infeasibles)
    return gen_loss_sequential, loss_sequential, w_sequential, z_sequential, infeasibles_sequential, predictions_sequential


def sequential_models_v2(config, Target, data, train_ID, test_ID, features, regularisation_level):
    """Experiment to determine effect of polynomial features  - v2, should be used"""
    # split into train/test data
    data_train = data[data['ID'].isin(train_ID)]
    data_test = data[data['ID'].isin(test_ID)]

    # Format x data to 24 hour format
    x_train = np.array(x_format_classifier(data_train, features, config.LT))
    x_test = np.array(x_format_classifier(data_test, features, config.LT))

    # dicts for saving results.
    predictions_sequential = {}
    mse_sequential = {}
    MSE = {}
    # Set up models in dict
    model_classes = {
        'LR_seq': LinearRegression(),
        'Ridge_seq': Ridge(regularisation_level[0]),
        'Lasso_seq': Lasso(regularisation_level[1]),
        'DT_seq': DecisionTreeRegressor(random_state=42),
        'RF_seq': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Train and predict for each target
    for targ in Target:
        y_train = np.array(x_format_classifier(data_train, [targ], config.LT))
        y_test = np.array(x_format_classifier(data_test, [targ], config.LT))

        # Loop over all models and train/predict
        for model_name, model in model_classes.items():

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            if model_name not in predictions_sequential:
                predictions_sequential[model_name] = {}

            predictions_sequential[model_name][targ] = y_pred

            if model_name not in mse_sequential:
                mse_sequential[model_name] = {}
                MSE[model_name] = {}
            mse_sequential[model_name][targ] = [mean_squared_error(
                y_test[i], y_pred[i]) for i in range(len(test_ID))]
            MSE[model_name][targ] = mean_squared_error(y_test, y_pred)

    return predictions_sequential, mse_sequential


def traders_relaxation_stochastic(config, train_ID, test_ID):
    """Stochatic optimisation problem for the relaxed traders optimisation problem"""
    DA_scenarios = []
    PW_scenarios = []
    for i in range(len(train_ID)):
        DA_scenarios.append(config.dataset[i]['DA_DK2'])
        PW_scenarios.append(config.dataset[i]['windpower'])

    # Problem
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    scenarios = train_ID
    n_scenarios = len(scenarios)
    prob = 1/n_scenarios

    model = gp.Model("Trader WF BESS")
    model.Params.LogToConsole = 0
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")

    model.setObjective(gp.quicksum(prob*gp.quicksum(-DA_scenarios[k][t] * (
        PW_scenarios[k][t] - p_ch[t] + p_dis[t]) for t in T) for k in scenarios), sense=GRB.MINIMIZE)

    c2 = model.addConstrs(
        (-p_ch[t] >= -P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (-p_dis[t] >= -P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                           1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((-SOC[t] >= -SOC_cap for t in T), name="Upper SOC")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        obj = model.ObjVal
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        res_stochastic = {'z': obj,
                          'pCH': p_ch_res,
                          'pDIS': p_dis_res,
                          'SOC': SOC_res,
                          'infeasibles': 0}

    z_realised_stochastic = []
    loss_stochastic = []

    for idx in range(len(test_ID)):
        # Calculate z_realised and loss for this sample
        DA_true = config.dataset[test_ID[idx]]['DA_DK2']
        PW_true = config.dataset[test_ID[idx]]['windpower']
        z_realised_val = - \
            np.sum(DA_true * (PW_true -
                   res_stochastic['pCH'] + res_stochastic['pDIS']))*config.scaler
        z_realised_stochastic.append(z_realised_val)
        loss_stochastic.append(
            z_realised_val - config.dataset[test_ID[idx]]['z']*config.scaler)

    gen_loss_stochastic = np.mean(loss_stochastic)

    w_stochastic = np.hstack([res_stochastic['pCH'], res_stochastic['pDIS'], np.ones(
        config.LT), res_stochastic['SOC']])

    return gen_loss_stochastic, loss_stochastic, w_stochastic


def EDP_stochastic(config, dataset, train_ID, scaler):
    """Stochatic optimisation problem for EDP"""
    scenarios = train_ID
    n_scenarios = len(scenarios)
    prob = 1/n_scenarios
    G = config.G
    T = config.T
    N = config.N
    LG = config.LG
    LT = config.LT
    LN = config.LN
    GN = config.GN
    NN = config.NN
    sus = config.sus
    cost_G = config.cost_G
    cost_G_up_reg = 1.2*cost_G
    cost_G_down_reg = 0.8*cost_G
    PD_scenarios = []
    for k in range(n_scenarios):
        PD_scenarios.append(np.array([dataset[k][f'P_D_N{n}'] for n in N]))
    # Two stage stochastic EDP
    model = gp.Model("Stochastic multihour nodal ED primal")
    model.Params.LogToConsole = 0
    p = model.addVars(G, T, vtype=GRB.CONTINUOUS, lb=0,
                      name="Generator production")
    if config.LN > 1:
        theta = model.addVars(
            scenarios, N, T, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="voltage angle")
    p_reg_up = model.addVars(
        scenarios, G, T, vtype=GRB.CONTINUOUS, lb=0, name="Generator up-regulation")
    p_reg_down = model.addVars(
        scenarios, G, T, vtype=GRB.CONTINUOUS, lb=0, name="Generator down-regulation")
    # Objective function
    model.setObjective(gp.quicksum(cost_G[g] * p[g, t] for g in G for t in T)
                       + gp.quicksum(prob * gp.quicksum(cost_G_up_reg[g] * p_reg_up[k, g, t]
                                                        - cost_G_down_reg[g] * p_reg_down[k, g, t] for g in G for t in T) for k in scenarios), sense=GRB.MINIMIZE)
    if config.LN == 1:
        balanceconstraints = model.addConstrs((gp.quicksum(p[g, t] + p_reg_up[k, g, t] - p_reg_down[k, g, t] for g in G)
                                               == PD_scenarios[k][0, t] for t in T for k in scenarios), name="Balance")
    else:
        balanceconstraints = model.addConstrs((gp.quicksum(p[g, t] + p_reg_up[k, g, t] - p_reg_down[k, g, t] for g in GN[n])
                                               - gp.quicksum(sus[n, m] * (theta[k, n, t] - theta[k, m, t]) for m in NN[n])
                                               == PD_scenarios[k][n, t] for n in N for t in T for k in scenarios), name="Balance")
        transmissionconstraints = model.addConstrs((sus[n, m] * (theta[k, n, t] - theta[k, m, t])
                                                    <= config.P_F[n, m] for k in scenarios for n in N for m in NN[n] for t in T), name="transmission capacity")
        thetazeroconstraints = model.addConstrs(
            (theta[k, 0, t] == 0 for k in scenarios for t in T), name="theta0")

    capacityconstraints = model.addConstrs(
        (p[g, t] <= config.P_G[g] for g in G for t in T), name="Upper bound capacity")
    capacityconstraintsreg = model.addConstrs((p[g, t] + p_reg_up[k, g, t] - p_reg_down[k, g, t]
                                               <= config.P_G[g] for g in G for k in scenarios for t in T), name="Upper bound capacity")
    constraintdown = model.addConstrs(
        (p_reg_down[k, g, t] <= p[g, t] for k in scenarios for g in G for t in T), name="down reg restriction")

    # Optimize
    model.optimize()

    model.write("test_stochastic_OPF.lp")

    if model.status == GRB.OPTIMAL:
        p_res = np.zeros((LG, LT))
        p_reg_up_res = np.zeros((len(scenarios), LG, LT))
        p_reg_down_res = np.zeros((len(scenarios), LG, LT))
        theta_res = np.zeros((len(scenarios), LN, LT))
        flow = np.zeros((len(scenarios), LN, LN, LT))
        for k in scenarios:
            for t in T:
                if config.LN > 1:
                    for n in N:
                        theta_res[k, n, t] = theta[k, n, t].x
                        for m in NN[n]:
                            flow[k, n, m, t] = sus[n, m] * \
                                (theta_res[k, n, t] - theta_res[k, m, t])
                            flow[k, m, n, t] = sus[m, n] * \
                                (theta_res[k, m, t] - theta_res[k, n, t])
                for g in G:
                    p_res[g, t] = p[g, t].x
                    for k in scenarios:
                        p_reg_up_res[k, g, t] = p_reg_up[k, g, t].x
                        p_reg_down_res[k, g, t] = p_reg_down[k, g, t].x
        obj = model.ObjVal * scaler
        print()
        print("Optimal solution found:")
        print(f"Total cost: {obj:.2f} DKK")
    else:
        print("Infeasible")
    return obj, p_res, p_reg_up_res, p_reg_down_res, flow


def EDP_bal(config, PD_scenarios, p, scaler):
    """Stochatic optimisation problem for EDP"""
    G = config.G
    T = config.T
    N = config.N
    LG = config.LG
    LT = config.LT
    LN = config.LN
    GN = config.GN
    NN = config.NN
    sus = config.sus
    cost_G = config.cost_G
    cost_G_up_reg = 1.2*cost_G
    cost_G_down_reg = 0.8*cost_G
    # Problem
    model = gp.Model("Stochastic multihour nodal ED primal")
    model.Params.LogToConsole = 0
    if config.LN > 1:
        theta = model.addVars(N, T, vtype=GRB.CONTINUOUS,
                              lb=-GRB.INFINITY, name="voltage angle")
    p_reg_up = model.addVars(G, T, vtype=GRB.CONTINUOUS,
                             lb=0, name="Generator up-regulation")
    p_reg_down = model.addVars(
        G, T, vtype=GRB.CONTINUOUS, lb=0, name="Generator down-regulation")
    # Objective function
    model.setObjective(gp.quicksum(cost_G[g] * p[g, t] + cost_G_up_reg[g] * p_reg_up[g, t] -
                       cost_G_down_reg[g] * p_reg_down[g, t] for g in G for t in T), sense=GRB.MINIMIZE)

    # Capacity constraints:

    if config.LN == 1:
        balanceconstraints = model.addConstrs((gp.quicksum(p[g, t] + p_reg_up[g, t] - p_reg_down[g, t] for g in G)
                                               == PD_scenarios[0, t] for t in T), name="Balance")
    else:
        balanceconstraints = model.addConstrs((gp.quicksum(p[g, t] + p_reg_up[g, t] - p_reg_down[g, t] for g in config.GN[n])
                                               - gp.quicksum(sus[n, m] * (theta[n, t] - theta[m, t]) for m in config.NN[n])
                                               == PD_scenarios[n, t] for n in N for t in T), name="Balance")
        transmissionconstraints = model.addConstrs((sus[n, m] * (theta[n, t] - theta[m, t])
                                                    <= config.P_F[n, m] for n in N for m in NN[n] for t in T), name="transmission capacity")
        thetazeroconstraints = model.addConstrs(
            (theta[0, t] == 0 for t in T), name="theta0")

    capacityconstraintsreg = model.addConstrs((p[g, t] + p_reg_up[g, t] - p_reg_down[g, t]
                                               <= config.P_G[g] for g in G for t in T), name="Upper bound capacity")
    constraintdown = model.addConstrs(
        (p_reg_down[g, t] <= p[g, t] for g in G for t in T), name="down reg restriction")

    # Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL:
        p_reg_up_res = np.zeros((len(G), len(T)))
        p_reg_down_res = np.zeros((len(G), len(T)))
        theta_res = np.zeros((len(N), len(T)))
        sigma1_res = np.zeros(len(T))
        flow = np.zeros((LN, LN, LT))
        for t in T:
            if config.LN > 1:
                sigma1_res[t] = thetazeroconstraints[t].Pi
                for n in N:
                    theta_res[n, t] = theta[n, t].x
                    for m in NN[n]:
                        flow[n, m, t] = sus[n, m] * \
                            (theta_res[n, t] - theta_res[m, t])
                        flow[m, n, t] = sus[m, n] * \
                            (theta_res[m, t] - theta_res[n, t])
            for g in G:
                p_reg_up_res[g, t] = p_reg_up[g, t].x
                p_reg_down_res[g, t] = p_reg_down[g, t].x
        obj = model.ObjVal * scaler
        DA_cost = sum(cost_G[g] * p[g, t] for g in G for t in T) * scaler
        RT_cost = sum(cost_G_up_reg[g] * p_reg_up_res[g, t] - cost_G_down_reg[g]
                      * p_reg_down_res[g, t] for g in G for t in T) * scaler
        print()
        print("Optimal solution found:")
        print(f"Total cost: {obj:.2f} DKK")
        print(f"DA cost: {DA_cost:.2f} DKK")
        print(f"RT cost: {RT_cost:.2f} DKK")
    else:
        print("Infeasible")

    return obj, p_reg_up_res, p_reg_down_res, flow, DA_cost, RT_cost


def balancing_test_EDP(config, model_names, test_ID, dataset, pgtest, p_sto, scaler):
    """Calculates the balance stage of EDP"""
    PD_test_scenarios = []
    for k in test_ID:
        PD_test_scenarios.append(
            np.array([dataset[k][f'P_D_N{n}'] for n in config.N]))
    Total_costs = {}
    DA_costs = {}
    RT_costs = {}
    p_dict = {}
    p_up_dict = {}
    p_down_dict = {}
    flows_dict = {}
    for model in model_names:

        Total_costs_array = np.zeros(len(test_ID))
        DA_costs_array = np.zeros(len(test_ID))
        RT_costs_array = np.zeros(len(test_ID))
        p_array = np.zeros((len(test_ID), config.LG, config.LT))
        p_up_array = np.zeros((len(test_ID), config.LG, config.LT))
        p_down_array = np.zeros((len(test_ID), config.LG, config.LT))
        flows_array = np.zeros((len(test_ID), config.LN, config.LN, config.LT))

        for k in range(len(test_ID)):
            if model == 'Stochastic':
                p = p_sto
            else:
                p = pgtest[model][k].T
            Tot_cost, p_up, p_down, flows, DA_cost, RT_cost = EDP_bal(
                config, PD_test_scenarios[k], p, scaler)
            Total_costs_array[k] = Tot_cost
            DA_costs_array[k] = DA_cost
            RT_costs_array[k] = RT_cost
            p_array[k] = p
            p_up_array[k] = p_up
            p_down_array[k] = p_down
            flows_array[k] = flows

        Total_costs[model] = Total_costs_array
        DA_costs[model] = DA_costs_array
        RT_costs[model] = RT_costs_array
        p_dict[model] = p_array
        p_up_dict[model] = p_up_array
        p_down_dict[model] = p_down_array
        flows_dict[model] = flows_array

    return Total_costs, DA_costs, RT_costs, p_dict, p_up_dict, p_down_dict, flows_dict


def stochastic_traders(config, train_ID):
    """Two-stage stochastic problem for traders"""
    DA_scenarios = []
    PW_scenarios = []
    BAL_scenarios = []
    for i in range(len(train_ID)):
        DA_scenarios.append(config.dataset[i]['DA_DK2'])
        PW_scenarios.append(config.dataset[i]['windpower'])
        BAL_scenarios.append(config.dataset[i]['DA_DK2']*0.9)

    # Problem
    T = config.T
    P_DIS_CH = config.P_DIS_CH
    SOC_init = config.SOC_init
    SOC_cap = config.SOC_cap
    eff = config.eff

    scenarios = train_ID
    n_scenarios = len(scenarios)
    prob = 1/n_scenarios
    K = range(n_scenarios)

    model = gp.Model("Stochastic model of Producers problem")
    model.Params.LogToConsole = 0
    p_w = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="windpower")
    p_ch = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS charge")
    p_dis = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS discharge")
    SOC = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="BESS SOC")
    Delta = model.addVars(K, T, vtype=GRB.CONTINUOUS,
                          lb=-GRB.INFINITY, name="Delta")

    model.setObjective(prob*gp.quicksum(DA_scenarios[k][t]*(
        p_w[t]+p_dis[t]) + BAL_scenarios[k][t]*Delta[k, t] for t in T for k in K), sense=GRB.MAXIMIZE)
    # Objective: Maximize expected profit from both DA and balancing market

    c1 = model.addConstrs((p_w[t]+p_ch[t] <= config.PW_cap for t in T))
    c2 = model.addConstrs(
        (p_ch[t] <= P_DIS_CH for t in T), name="Upper ch")
    c3 = model.addConstrs(
        (p_dis[t] <= P_DIS_CH for t in T), name="Upper dis")
    c4 = model.addConstr(
        (SOC[0] == SOC_init + eff * p_ch[0] - 1/eff * p_dis[0]), name="SOC 1")
    c5 = model.addConstrs((SOC[t] == SOC[t-1] + eff * p_ch[t] -
                           1/eff * p_dis[t] for t in range(1, len(T))), name="SOC t")
    c6 = model.addConstrs((SOC[t] <= SOC_cap for t in T), name="Upper SOC")

    c7 = model.addConstrs((Delta[k, t] == PW_scenarios[k][t] - (p_w[t]+p_ch[t])
                          for k in K for t in T), name="Delta constraint")

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Optimal")
        obj = model.ObjVal
        print(obj)
        p_w_res = np.array([p_w[t].x for t in T])
        p_ch_res = np.array([p_ch[t].x for t in T])
        p_dis_res = np.array([p_dis[t].x for t in T])
        SOC_res = np.array([SOC[t].x for t in T])
        Delta_res = []
        for k in K:
            Delta_res.append(np.array([Delta[k, t].x for t in T]))
        res_stochastic = {'z': obj,
                          'p_w': p_w_res,
                          'pCH': p_ch_res,
                          'pDIS': p_dis_res,
                          'SOC': SOC_res,
                          'Delta_res': Delta_res,
                          'infeasibles': 0}
    return res_stochastic


###################### PLOTS #####################


def decision_plot(w_preds: dict, variables: list, model_names: list, idx: int):
    """Plots the decision variables for different models.

    Arguments:
        w_preds: Dictionary of predicted weights for each model.
        variables: List of variable names to plot.
        model_names: List of model names.
        idx: Index for selecting the specific prediction.

    """
    num_vars = len(variables)
    fig, axs = plt.subplots(num_vars, 1, figsize=(12, 6), sharex=True)

    x_values = np.arange(0, 24)

    # Prepare data for each model
    for model_name in model_names:
        df = w_var_translator(w_preds[model_name][idx], variables)

        # print("df:", df)

        for i, var in enumerate(variables):
            axs[i].step(x_values, df[var].values, label=model_name)

    # Set titles and labels
    for i, var in enumerate(variables):
        axs[i].set_title(var)
        axs[i].set_ylabel("Power [-]")
        axs[i].grid(alpha=0.4)
        # Set custom x-ticks
        axs[i].set_xticks(x_values)  # Set the x-ticks to 1 and 24

    # Set x-label for the bottom subplot
    axs[-1].set_xlabel("Hours")

    # Adjust layout and legend
    plt.tight_layout()
    plt.legend(loc="upper center", bbox_to_anchor=(
        0.5, -1.25), fancybox=True, ncol=len(model_names))

    plt.show()


def plot_subplots_equality2(results_list, names_list, LT, figsize=(20, 10)):
    """Plots results for traders"""
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda$", "$-\\lambda$",
              "$-\\lambda * PW$", "0", "PW"]
    data_ranges = [
        (0, len(results_list[0])),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (2*LT, 3*LT)]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            if i == 5:  # Special case for "PW" plot
                ax.plot(results_list[0][start:end] / results_list[0][LT:2*LT],
                        label=names_list[0], color=c_list[0])
                ax.plot(results_list[1][start:end] / results_list[1][LT:2*LT],
                        label=names_list[1], color=c_list[1])
                ax.plot(results_list[2][start:end] / results_list[2][LT:2*LT],
                        label=names_list[2], color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color='grey')
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color='black')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_EDP_nodal(model_names, plotlist, plotdata, plot_sample, plotcategory, plotX, plotXname, plotindex):
    """Plots EDP nodal results"""
    if len(plotlist) > 1:
        fig, axs = plt.subplots(
            len(plotlist), 1, figsize=(12, 8+len(plotlist)))
    else:
        plt.figure(figsize=(12, 4))

    for i, model_name in enumerate(model_names):
        if len(plotlist) > 1:
            for ki, k in enumerate(plotlist):
                if plotindex == 0:
                    axs[ki].step(np.arange(0, plotX), plotdata[model_name]
                                 [plot_sample][k], label=model_name)
                if plotindex == 1:
                    axs[ki].step(np.arange(0, plotX), plotdata[model_name]
                                 [plot_sample][:, k], label=model_name)
                axs[ki].set_title(plotcategory +
                                  str(k) + " and sample " + str(plot_sample))
                axs[ki].set_ylabel("Power [-]")
                axs[ki].grid(alpha=0.4)
                axs[ki].set_xticks(np.arange(0, plotX))
            axs[-1].set_xlabel(plotXname)
        else:
            if plotindex == 0:
                plt.step(np.arange(0, plotX), plotdata[model_name]
                         [plot_sample][plotlist[0]], label=model_name)
            if plotindex == 1:
                plt.step(np.arange(0, plotX), plotdata[model_name]
                         [plot_sample][:, plotlist[0]], label=model_name)
            plt.title(plotcategory +
                      str(plotlist[0]) + " and sample " + str(plot_sample))
            plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.legend(loc="lower center", bbox_to_anchor=(
        0.5, -0.4), fancybox=True, ncol=len(model_names))
    plt.show()


xlabel = "Number of training samples"


def barplot_gen_loss(gen_loss):
    """Creates bar plots"""
    # Create the bar plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.bar(gen_loss.keys(), gen_loss.values(), color='skyblue')

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Generalisation Loss')
    plt.xticks(rotation=90)  # Rotate x labels for better readability

    # Display the plot
    plt.tight_layout()
    plt.show()


def loss_boxplot_experiment(losses, infeasibles, levels, xlabel, outliers=True, n_samples=True, fig_size=(12, 6), box_width=0.8, save_plot_name=None):
    """Creates boxplots for robustness experiments"""
    total = []
    # Prepare data for each level
    for level in levels:
        temp_loss = pd.DataFrame(losses[level])
        temp_infeasibles = pd.DataFrame(infeasibles[level])

        # Replace infeasible losses with NaN
        temp_loss = temp_loss.where(temp_infeasibles == 0, np.nan)
        temp_loss['Label'] = level  # Add level as a label
        total.append(temp_loss)

    total = pd.concat(total, ignore_index=True)

    # Melt the DataFrame for seaborn compatibility
    df_melt = pd.melt(total, id_vars='Label')

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    sns.boxplot(x='Label', y='value', hue='variable', data=df_melt,
                width=box_width, showfliers=outliers, ax=ax)

    # Add a title and labels
    plt.title('SPO Loss Distribution', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('SPO loss [DKK]', fontsize=12)

    # Calculate observation counts for each level and variable combination, excluding NaNs
    counts = df_melt.groupby(['Label', 'variable'])['value'].count()

    # Calculate the medians for each group and variable combination
    medians = df_melt.groupby(['Label', 'variable'])['value'].median()
    if n_samples == True:
        # # Loop through each unique label and its corresponding boxes
        for tick, label in enumerate(ax.get_xticklabels()):
            level = label.get_text()  # Get the level name
            print(levels[tick])
            print(level)
            # Loop through each variable (hue)
            for j, hue in enumerate(df_melt['variable'].unique()):
                # Create a tuple to access the MultiIndex
                if isinstance(levels[tick], int):
                    key = (int(level), hue)
                elif isinstance(levels[tick], float):
                    key = (float(level), hue)
                num = counts.get(key, 0)  # Get the count, default to 0
                median_value = medians.get(key, np.nan)  # Get the median value

                # Ensure the median is a finite number
                # if np.isfinite(median_value):
                # Calculate positions
                num_hues = len(df_melt['variable'].unique())
                x_pos = tick + (j - (num_hues - 1) / 2) * \
                    (box_width / num_hues)
                # Position text above the median
                # A small offset above the median
                pos_y = median_value + (median_value * 0.02)

                # Add observation count text
                ax.text(x_pos, pos_y, f'n: {num}',
                        horizontalalignment='center', size='small', color='white', weight='semibold', zorder=100)

    # Show the legend
    plt.legend(title='Model')
    plt.grid(True, which='both', axis='y', alpha=0.3, zorder=-100)

    # Clean the plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.set_facecolor(color='white')
    plt.tight_layout()

    # Show the legend
    plt.legend(title='Model')
    plt.grid(True, which='both', axis='y', alpha=0.3, zorder=-100)

    # Clean the plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.set_facecolor(color='white')
    plt.tight_layout()

    plt.show()


############################# PLOTS ####################################


def plot_subplots_lagrangian(c_spo, c_lr, c_test, LT, figsize=(20, 10)):
    """plots results for traders on lagrangian"""
    c_list = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda - \\mu$", "$\\mu$",
              "$\\lambda$", "Zero vector", "Pw"]
    data_ranges = [
        (0, len(c_spo)),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (4*LT, 5*LT)
    ]

    # Lines and their respective labels
    lines_labels = [
        (c_spo, "SPO", c_list[0]),
        (c_lr, "Lin. Reg.", c_list[1]),
        (c_test, "Target", c_list[2])
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for data, label, color in lines_labels:
            if i == 5:  # Special case for "PW" plot, different data processing
                ax.plot(c_spo[start:end] / (-1 * c_spo[LT:2*LT]),
                        label=label, color=c_list[0])
                ax.plot(c_lr[start:end] / (-1 * c_lr[LT:2*LT]),
                        label=label, color=c_list[1])
                ax.plot(c_test[start:end] / (-1 * c_test[LT:2*LT]),
                        label=label, color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=color)

        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color=c_list[3])
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color=c_list[4])

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(c_list))
    plt.show()


def plot_subplots_lagrangian2(results_list, names_list, LT, figsize=(20, 10)):
    """plots results for traders on lagrangian v2"""
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda - \\mu$", "$\\mu$",
              "$\\lambda$", "Zero vector", "Pw"]
    data_ranges = [
        (0, len(results_list[0])),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (4*LT, 5*LT)
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            if i == 5:  # Special case for "PW" plot, different data processing
                ax.plot(results_list[0][start:end] / (-1 * results_list[0][LT:2*LT]),
                        label=names_list[0], color=c_list[0])
                ax.plot(results_list[1][start:end] / (-1 * results_list[1][LT:2*LT]),
                        label=names_list[1], color=c_list[1])
                ax.plot(results_list[2][start:end] / (-1 * results_list[2][LT:2*LT]),
                        label=names_list[2], color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color='grey')
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color='black')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_subplots_dual(c_spo, c_lr, c_test, LT, figsize=(20, 10)):

    c_list = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "Pw", "P^Ch/P^Dis",
              "$P^Ch/P^Dis", "SOC", "SOC_init"]

    data_ranges = [
        (0, len(c_spo)),
        (0, LT+1),
        (LT+1, 2*LT+1),
        (2*LT+1, 3*LT+1),
        (3*LT+1, 4*LT+1),
        (4*LT+1, 5*LT+1)
    ]

    lines_labels = [
        (c_spo, "SPO", c_list[0]),
        (c_lr, "Lin. Reg.", c_list[1]),
        (c_test, "Target", c_list[2])
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for data, label, color in lines_labels:
            ax.plot(data[start:end], label=label, color=color)
        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Feasibility limit", color=c_list[4])

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(c_list))
    plt.show()


def plot_subplots_dual2(results_list, names_list, LT, figsize=(20, 10)):
    """plots results for traders on dual formulation"""
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "Pw", "P^Ch/P^Dis",
              "$P^Ch/P^Dis$", "SOC", "SOC_init"]

    data_ranges = [
        (0, len(results_list[0])),
        (0, LT+1),
        (LT+1, 2*LT+1),
        (2*LT+1, 3*LT+1),
        (3*LT+1, 4*LT+1),
        (4*LT+1, 5*LT+1)
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 1 or i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Feasibility limit", color='grey')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_subplots_equality(c_spo, c_lr, c_test, LT, figsize=(20, 10)):
    """plots results for traders on reformulation"""
    c_list = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda$", "$-\\lambda$",
              "$-\\lambda * PW$", "0", "PW"]
    data_ranges = [
        (0, len(c_spo)),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (2*LT, 3*LT)]

    # Lines and their respective labels
    lines_labels = [
        (c_spo, "SPO", c_list[0]),
        (c_lr, "Lin. Reg.", c_list[1]),
        (c_test, "Target", c_list[2])
    ]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for data, label, color in lines_labels:
            if i == 5:  # Special case for "PW" plot, different data processing
                ax.plot(c_spo[start:end] / (c_spo[LT:2*LT]),
                        label=label, color=c_list[0])
                ax.plot(c_lr[start:end] / (c_lr[LT:2*LT]),
                        label=label, color=c_list[1])
                ax.plot(c_test[start:end] / (c_test[LT:2*LT]),
                        label=label, color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=color)

        if i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color=c_list[3])
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color=c_list[4])

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(c_list))
    plt.show()


def plot_subplots_equality2(results_list, names_list, LT, figsize=(20, 10)):
    """plots results for traders on reformulation v2"""
    c_list = [f'C{i}' for i in range(len(results_list))]

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    titles = ["C-vector", "$\\lambda$", "$-\\lambda$",
              "$-\\lambda * PW$", "0", "PW"]
    data_ranges = [
        (0, len(results_list[0])),
        (0, LT),
        (LT, 2*LT),
        (2*LT, 3*LT),
        (3*LT, 4*LT),
        (2*LT, 3*LT)]

    for i, ax in enumerate(axs.flatten()):
        start, end = data_ranges[i]
        ax.set_title(titles[i])

        for j, (data, label) in enumerate(zip(results_list, names_list)):
            if i == 5:  # Special case for "PW" plot
                ax.plot(results_list[0][start:end] / results_list[0][LT:2*LT],
                        label=names_list[0], color=c_list[0])
                ax.plot(results_list[1][start:end] / results_list[1][LT:2*LT],
                        label=names_list[1], color=c_list[1])
                ax.plot(results_list[2][start:end] / results_list[2][LT:2*LT],
                        label=names_list[2], color=c_list[2])
            else:
                ax.plot(data[start:end], label=label, color=c_list[j])

        if i == 4:  # For "Lambda - mu" and "Zero" plots
            ax.plot(np.zeros(LT), linestyle="--",
                    label="Theoretical Target", color='grey')
        elif i == 5:  # For "PW" plot
            ax.plot(np.zeros(LT), linestyle='--',
                    label="Feasibility Limit", color='black')

        if i != 0:  # Add x-ticks to all plots except the first one
            ax.set_xticks(np.arange(0, LT, 2))

        ax.grid(alpha=0.4)

    # Extract all lines and labels from the subplots
    lines, labels = [], []
    for ax in axs.flatten():
        for line in ax.get_lines():
            if line.get_label() not in labels:
                lines.append(line)
                labels.append(line.get_label())

    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(
        0.5, 0.1), fancybox=True, ncol=len(results_list))
    plt.show()


def plot_cluster(df, n_clusters, cluster_target, cluster_center, LT, save_name=None):
    """plots cluster if 1d target"""
    c_list = [f'C{i}' for i in range(n_clusters)]
    plt.figure(figsize=(12, 8))
    for idx in range(n_clusters):
        cluster_df = df[df['Cluster'] == idx]
        plt.plot(cluster_center[idx], color=c_list[idx], label=f"Cluster {idx+1}", linewidth=3, path_effects=[
                 pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
        for id in cluster_df.ID.unique():
            plt.plot(cluster_df[cluster_df['ID'] == id]
                     [cluster_target].values, color=c_list[idx], alpha=0.3, zorder=0)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xlabel("Hours")
    plt.ylabel("Value [-]")
    if save_name is not None:
        plt.savefig(f'{save_name}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.show()


def plot_cluster_multitarget(df, n_clusters, cluster_targets, cluster_center, LT, save_name=None):
    """plots cluster if 2d target"""
    c_list = [f'C{i}' for i in range(n_clusters)]

    # Combined plot for each target
    for jdx, target in enumerate(cluster_targets):
        plt.figure(figsize=(12, 8))

        for idx in range(n_clusters):
            cluster_df = df[df['Cluster'] == idx]
            cluster_mean = cluster_center[idx].flatten()[jdx*LT:(jdx+1)*LT]

            plt.plot(cluster_mean, color=c_list[idx], label=f"Cluster {idx+1} - {target}",
                     linewidth=3, path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])

            for id in cluster_df.ID.unique():
                plt.plot(cluster_df[cluster_df['ID'] == id][target].values,
                         color=c_list[idx], alpha=0.3, zorder=0)

        plt.legend()
        plt.grid(alpha=0.4)
        plt.xlabel("Hours")
        plt.ylabel(f"{target} [-]")
        plt.title(f"Clustered {target}")
    if save_name is not None:
        plt.savefig(f'{save_name}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.show()


def loss_boxplot_general(loss_dict, infeasibles, mse=False, save_plot_name=None):
    """Creates loss boxplots"""
    infeasibles = pd.DataFrame(infeasibles)

    loss_spo = {model: results['SPO loss']
                for model, results in loss_dict.items()}

    temp_spo = pd.DataFrame(loss_spo)
    temp_no_infeasibles_spo = temp_spo.where(infeasibles == 0, np.nan)

    temp_melt = pd.melt(temp_no_infeasibles_spo)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(x='variable', y='value', data=temp_melt, width=0.5)
    plt.title('SPO Loss Distribution', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('SPO loss [DKK]', fontsize=12)

    # Show the legend
    plt.grid(True, which='both', axis='y', alpha=0.3, zorder=-100)

    # Show the plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.set_facecolor(color='white')
    plt.tight_layout()

    if save_plot_name is not None:
        fig.savefig(f"{save_plot_name}_spo.png", dpi=300)
    plt.show()
    if mse == True:
        loss_mse = {model: results['MSE']
                    for model, results in loss_dict.items()}
        temp_mse = pd.DataFrame(loss_mse)
        temp_no_infeasibles_mse = temp_mse.where(infeasibles == 0, np.nan)
        temp_melt_mse = pd.melt(temp_no_infeasibles_mse)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.boxplot(x='variable', y='value', data=temp_melt_mse, width=0.5)
        plt.title('MSE Loss Distribution', fontsize=14)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('MSE loss [-]', fontsize=12)

        # Show the legend
        plt.grid(True, which='both', axis='y', alpha=0.3, zorder=-100)

        # Show the plot
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.set_facecolor(color='white')
        plt.tight_layout()

        if save_plot_name is not None:
            fig.savefig(f"{save_plot_name}_mse.png", dpi=300)
        plt.show()


def boxplot_spo_loss(loss_dict, infeasibles, outliers=True, n_samples=True, fig_size=(12, 6), box_width=0.8, save_plot_name=None):
    """Creates loss boxplots for SPO loss"""
    # Convert infeasibles to DataFrame
    infeasibles = pd.DataFrame(infeasibles)

    # Convert loss_dict to DataFrame
    temp_spo = pd.DataFrame(loss_dict)

    # Replace infeasible values with NaN
    temp_no_infeasibles_spo = temp_spo.where(infeasibles == 0, np.nan)

    # Melt the DataFrame for seaborn compatibility
    temp_melt = pd.melt(temp_no_infeasibles_spo)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    sns.boxplot(x='variable', y='value', hue='variable', data=temp_melt,
                width=box_width, showfliers=outliers, ax=ax)

    # plt.title('SPO Loss Distribution', fontsize=14)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('SPO loss [DKK]', fontsize=10)
    if n_samples == True:
        # Calculate the medians and observation counts per group, excluding NaNs
        medians = temp_melt.groupby('variable')['value'].median()
        nobs = temp_melt.groupby('variable')[
            'value'].count()  # Count non-NaN values

        # Prepare the observation count text
        nobs_text = nobs.apply(lambda x: f'n: {x}')

        # Loop through each x-axis tick label (box)
        for tick, label in enumerate(ax.get_xticklabels()):
            variable = label.get_text()  # Get the variable name from the label

            # Get the median and count for the current variable
            med_val = medians[variable]
            num = nobs_text[variable]

            # Adjust position to display the text above the boxplot median
            ax.text(tick, med_val + (med_val * 0.05), num,
                    horizontalalignment='center', size='small', color='white', weight='semibold')

    # Show the grid and clean the plot
    plt.grid(True, which='both', axis='y', alpha=0.3, zorder=-100)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.set_facecolor(color='white')
    plt.tight_layout()

    # Save the plot if a filename is provided
    if save_plot_name is not None:
        fig.savefig(f"{save_plot_name}_spo.png", dpi=300)
    plt.show()


def RT_market_producers(config, data, test_ID, predictions_spo, predictions_sequential):
    """Calculates the RT market for producers"""
    RT = pd.read_csv("RT_price.csv")
    RT = RT.rename(columns={"HourUTC": "timestamp"})
    RT['timestamp'] = pd.to_datetime(RT['timestamp'])
    RT = RT.set_index("timestamp", drop=True)
    RT = RT*7.45
    data['RT_DK2'] = RT.loc[data.index, 'RT_DK2']

    Delta_tot = {}
    RT_tot = {}
    Balance_cost_tot = {}

    for model in predictions_spo.keys():
        Delta = []
        RT = []
        Balance_cost = []

        for i in range(len(test_ID)):
            PW_pred = predictions_spo[model][i][2 *
                                                24:3*24]/predictions_spo[model][i][24:2*24]
            PW_true = config.dataset[test_ID[i]]['windpower']

            Delta_val = (PW_true - PW_pred)*config.base_scaler
            RT_val = data[data['ID'] == test_ID[i]].RT_DK2
            Balance_cost_val = np.sum(Delta_val*RT_val)

            Delta.append(Delta_val)
            RT.append(RT_val)
            Balance_cost.append(Balance_cost_val)

        Delta_tot[model] = Delta
        RT_tot[model] = RT
        Balance_cost_tot[model] = Balance_cost

    for model in predictions_sequential.keys():
        Delta = []
        RT = []
        Balance_cost = []
        for i in range(len(test_ID)):
            PW_pred = predictions_sequential[model]['windpower'][i]
            PW_true = config.dataset[test_ID[i]]['windpower']
            Delta_val = (PW_true - PW_pred)*config.base_scaler
            RT_val = data[data['ID'] == test_ID[i]].RT_DK2
            Balance_cost_val = np.sum(Delta_val*RT_val)

            Delta.append(Delta_val)
            RT.append(RT_val)
            Balance_cost.append(Balance_cost_val)

        Delta_tot[model] = Delta
        RT_tot[model] = RT
        Balance_cost_tot[model] = Balance_cost

    return Delta_tot, RT_tot, Balance_cost_tot


def traders_feasibility_check(config, w_prediction, check_decision, test_ID):
    """Checks feasibility of solution for producers."""
    infeasible_prediction = {}

    for model in w_prediction.keys():
        infeasible = []
        for i in range(len(test_ID)):
            if check_decision == "pw_pch":
                pDA = w_prediction[model][i][:24] + \
                    w_prediction[model][i][2*24:3*24]
                PW_true = config.dataset[test_ID[i]
                                         ]['windpower']*config.base_scaler
            elif check_decision == "pch":
                pDA = w_prediction[model][i][:24]
                PW_true = config.dataset[test_ID[i]
                                         ]['windpower']*config.base_scaler
            if any(pDA > PW_true):
                infeasible.append(1)
            else:
                infeasible.append(0)
        infeasible_prediction[model] = np.array(infeasible)
    return infeasible_prediction
