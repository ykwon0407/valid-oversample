import numpy as np
from multiprocessing import Pool
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from utils import evaluate_prediction


def cvxopt_solve_minmax(X, Y, lambda_n=0.4):
    index_0, index_1 = np.where(Y == 0)[0], np.where(Y == 1)[0]
    X1, X2 = X[index_0], X[index_1]

    n1, p = X1.shape
    n2, p = X2.shape
    n = n1+n2

    delta = np.mean(X1, axis=0)-np.mean(X2, axis=0)
    sigma1 = np.cov(X1.T)*(n1-1)/n1
    sigma2 = np.cov(X2.T)*(n2-1)/n2
    sigma = (n1*sigma1+n2*sigma2)/(n1+n2) + ((np.log(p)/n)**(0.5))*np.eye(p)

    vec_constraint1 = np.repeat(0,2*p)
    mat_constraint1 = np.zeros((2*p,2*p))
    for i in range(2*p):
        for j in range(2*p):
            if(abs(i-j) % p == 0):
                if i >= p and j >= p :
                    mat_constraint1[i,j] = +1
                else:
                    mat_constraint1[i,j] = -1

    mat_constraint2_part1 = np.concatenate((np.zeros((p,p)), -sigma), axis=1)
    mat_constraint2_part2 = np.concatenate((np.zeros((p,p)), sigma), axis=1)
    mat_constraint2 = np.concatenate((mat_constraint2_part1, mat_constraint2_part2))

    c = np.repeat([1.0,0.0], p)
    A = np.concatenate((mat_constraint1, mat_constraint2))
    vec_constraint2 = np.concatenate((lambda_n - delta, lambda_n + delta))
    b = np.concatenate((vec_constraint1, vec_constraint2))

    A_ = matrix(A)
    c_ = matrix(c)
    b_ = matrix(b)

    sol = solvers.lp(c_, A_, b_, solver='cvxopt_glpk')
    beta = np.array(sol['x']).reshape(-1)[p:2*p]
    return beta

class VO_LPD(object):
    '''
    Valid oversampling based on kernel LPD
    '''
    def __init__(self, X, Y, spec_limit=0.05, sen_limit=0.03):
        np.random.seed(1004)
        self.X_raw, self.Y_raw = X, Y
        self.X, self.Y = self.X_raw, self.Y_raw
        self.spec_limit = spec_limit
        self.sen_limit = sen_limit
        self.p_sugg = 0.5
        self.lambda_list = np.linspace(1e-4, 1e-2, 6)
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)

    def suggest_valid_prop(self):
        self.lambda_lpd = self.get_lambda()
        self.beta_lpd = cvxopt_solve_minmax(self.X_raw, self.Y_raw,
                                             lambda_n=self.lambda_lpd)
        self.p_sugg = self.calculate_valid_prop_core() 
        return np.round(self.p_sugg, 4)


    def get_lambda(self):
        N = len(self.lambda_list)
        result = np.zeros((N,1))
        for j in range(N):
            result[j,0] = self.get_lambda_core(j)        

        sum_cv = np.array(result).reshape(N,1)
        lambda_list = self.lambda_list.reshape(N,1)
        res = np.concatenate((lambda_list, sum_cv), axis=1)
        
        index = np.argmax(res[:,1])
        lambda_n = res[index][0]
            
        return lambda_n 

    def get_lambda_core(self, k):
        sum_cv = 0.0
        for tr, te in self.skf.split(self.X, self.Y):
            Y_tr, Y_te = self.Y[tr], self.Y[te]
            X_tr, X_te = self.X[tr], self.X[te]
            p = np.mean(Y_tr)
            X_tr_1, X_tr_2 = X_tr[np.where(Y_tr == 0)], X_tr[np.where(Y_tr == 1)]
            X_te_1, X_te_2 = X_te[np.where(Y_te == 0)], X_te[np.where(Y_te == 1)]
            beta = cvxopt_solve_minmax(X=X_tr, Y=Y_tr, lambda_n=self.lambda_list[k])
            mu = (np.mean(X_tr_1, axis=0)+np.mean(X_tr_2, axis=0))/2.0
            Y_te_pred = ((X_te-mu).dot(beta) < np.log(p/(1-p))) + 0.0
            sum_cv += (np.sum(Y_te == Y_te_pred) + 0.0)/len(self.X)
        return sum_cv

    def calculate_valid_prop_core(self):
        bw = lambda n : (n * (1 + 2) / 4.)**(-1. / (1 + 4)) 
        logistic = lambda x: 1/(1+np.exp(-x))

        # Calculate Delta
        index_0, index_1 = np.where(self.Y_raw == 0)[0], np.where(self.Y_raw == 1)[0]
        X_train_0, X_train_1 = self.X_raw[index_0], self.X_raw[index_1]

        n0, p = X_train_0.shape
        n1, p = X_train_1.shape
        n = n0 + n1

        mu_0, mu_1 = np.mean(X_train_0, axis=0), np.mean(X_train_1, axis=0)
        W = (self.X_raw-(mu_0+mu_1)/2.).dot(self.beta_lpd).reshape(-1,1)

        kde_0 = KernelDensity(kernel='gaussian', bandwidth=bw(n0)).fit((W[index_0]))
        kde_1 = KernelDensity(kernel='gaussian', bandwidth=bw(n1)).fit((W[index_1]))
        bayes = lambda t : logistic(kde_0.score_samples(t) - kde_1.score_samples(t)) 
        bayes_0 = np.array([bayes([t]) for t in W[index_0]]).reshape(-1,1)
        bayes_1 = np.array([bayes([t]) for t in W[index_1]]).reshape(-1,1)

        # since bayes_1 < bayes_0
        upper_bound_p = np.percentile(bayes_0, self.spec_limit*100) # spec
        lower_bound_p = np.percentile(bayes_1, (1.-self.sen_limit)*100) # sen
        p_sugg = self.find_feasible_set(upper_bound_p, lower_bound_p, bayes_0, bayes_1)

        return p_sugg 

    def find_feasible_set(self, A, B, bayes_0, bayes_1, N_rpt=400):
        # feasible test function
        feasible_test = False
        if A >= B : feasible_test = True; pass
        else: print('Not feasible, we choose 0.5')
        
        if feasible_test is True:
            result = np.zeros((N_rpt, 3))
            for idx, x in enumerate(np.linspace(A, B, N_rpt)):
                result[idx] = np.array([x, np.mean(bayes_0 < x), np.mean(bayes_1 > x)]) # prob, spec, sen
            p_sugg = result[(np.argmin(np.max(result[:,1:], axis=1))),0]
            return p_sugg
        else:
            return 0.5

    '''
    Functions used after oversampling
    '''      
    def oversample_data(self):
        np.random.seed(1)
        N_sample = int(len(self.Y_raw)*(self.p_sugg-0.1)/(1-self.p_sugg))
        idx = np.random.choice(np.where(self.Y_raw==1)[0], N_sample)
        X_tmp, Y_tmp = self.X_raw[idx], self.Y_raw[idx]
        self.X_over, self.Y_over = np.vstack((self.X_raw, X_tmp)), np.hstack((self.Y_raw, Y_tmp)) 
        self.X, self.Y = self.X_over, self.Y_over

    def calculate_performance(self, X_te, Y_te):
        bw = lambda n : (n * (1 + 2) / 4.)**(-1. / (1 + 4)) 
        logistic = lambda x: 1/(1+np.exp(-x))

        lambda_lpd_over = self.get_lambda()
        beta_lpd_over = cvxopt_solve_minmax(self.X_over, self.Y_over, lambda_n=lambda_lpd_over)

        # Calculate Delta
        index_0, index_1 = np.where(self.Y_over == 0)[0], np.where(self.Y_over == 1)[0]
        X_train_0, X_train_1 = self.X_over[index_0], self.X_over[index_1]

        n0, p = X_train_0.shape
        n1, p = X_train_1.shape
        n = n0 + n1

        mu_0, mu_1 = np.mean(X_train_0, axis=0), np.mean(X_train_1, axis=0)
        W_over_tr = (self.X_over-(mu_0+mu_1)/2.).dot(beta_lpd_over).reshape(-1,1)
        W_te = (X_te-(mu_0+mu_1)/2.).dot(beta_lpd_over).reshape(-1,1)

        # logistic with log ratio
        kde_0 = KernelDensity(kernel='gaussian', bandwidth=bw(n0)).fit((W_over_tr[index_0]))
        kde_1 = KernelDensity(kernel='gaussian', bandwidth=bw(n1)).fit((W_over_tr[index_1])) 
        bayes_over = lambda t : logistic(kde_0.score_samples(t) - kde_1.score_samples(t)) 

        # prediction
        bayes_pred = np.array([bayes_over([t]) for t in W_te])
        Y_te_pred = ((bayes_pred <= self.p_sugg) + 0.0).reshape(-1)

        sensitivity, specificity, accuracy, average_accuracy = evaluate_prediction(Y_te, Y_te_pred)
        print('Test set: Sensitivity: {:.4f}, Specificity: {:.4f}, Accuracy: {:.4f}, AA: {:.4f}'.format(
            sensitivity, specificity, accuracy, average_accuracy))








