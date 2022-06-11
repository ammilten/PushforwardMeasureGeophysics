import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
import scipy as sp
import scipy.stats as st

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from SeisSim import LayerSeismic

def prepare_M(theta):
    return LayerSeismic(rhob=theta[0], Vb=theta[1], rhol=theta[2], Vl=theta[3], h=theta[4], dh=theta[5], fM=theta[6], T=0.8, dt=0.004, outfile=None) # From real. 9000

def get_z(Mobs):
    Mobs.run_full()
    return Mobs.seismic

def normalize_data(p):
    pmin = np.min(p,axis=0)
    pmax = np.max(p,axis=0)
    pnorm = (p-pmin)/(pmax-pmin)
    
    nnans = np.sum(np.isnan(pnorm),axis=0)
    const_inds = np.where(nnans == p.shape[0])[0]
    pnorm[:,const_inds] = np.ones((p.shape[0],1))
    return pnorm, pmin, pmax

def unnormalize_data(pnorm, pmin, pmax):
    return pnorm * (pmax-pmin) + pmin

def import_mc_data(path_to_folder):
    '''
    This function takes a path to the folder containing the Monte Carlo simulation and imports the data
    
    Inputs:
        path_to_folder: string of path to the to folder containing the Monte Carlo simulation data

    Outputs:
        D: [SxD] matrix of data realizations in each row
        t: [Dx1] time vector
        inputs:
        varnames: 
    '''
    pfile = path_to_folder + 'params.dat'


    files = [name for name in os.listdir(path_to_folder) if (os.path.isfile(path_to_folder+name) and (name != "params.dat"))]

    Nreal = len(files)
    data = np.loadtxt(path_to_folder+files[0], delimiter=",")
    D = np.zeros((Nreal,data.shape[0]))
    inds = np.zeros(Nreal,dtype=int)
    i = 0
    for f in files:
        data = np.loadtxt(path_to_folder+f, delimiter=",")
        D[i,:] = data[:,1]
        f2 = f.split('_')[1]
        inds[i] = np.int(f2.split('.')[0])
        i = i+1
    t = data[:,0]
    fullinput = pd.read_csv(pfile,skiprows=11) # 11 is the numer of header rows in the SeismicLayerMonteCarlo input file
    inputs = fullinput.values[inds,:]
    varnames = list(fullinput.columns)

    return D, t, inputs, varnames

def load_priors(MCfolder):

    if MCfolder.endswith('/'):
        pfile = MCfolder + 'params.dat'
    else:
        pfile = MCfolder + '/params.dat'
    if not os.path.exists(pfile):
        print("Error: " + pfile + " does not exist. Aborting")
        sys.exit(2)

    with open(pfile,"r") as pf: # Might need to re-do to account for whitespace
        comm = '#'
        sep = ','

        T = float(pf.readline().split(comm,1)[0])
        dt = float(pf.readline().split(comm,1)[0])

        rhobln = pf.readline().split(comm,1)[0].split(sep)
        rhob = [float(rhobln[0]), float(rhobln[1])]

        Vbln = pf.readline().split(comm,1)[0].split(sep)
        Vb = [float(Vbln[0]), float(Vbln[1])]

        rholln = pf.readline().split(comm,1)[0].split(sep)
        rhol = [float(rholln[0]), float(rholln[1])]

        Vlln = pf.readline().split(comm,1)[0].split(sep)
        Vl = [float(Vlln[0]), float(Vlln[1])]

        hln = pf.readline().split(comm,1)[0].split(sep)
        h = [float(hln[0]), float(hln[1])]

        dhln = pf.readline().split(comm,1)[0].split(sep)
        dh = [float(dhln[0]), float(dhln[1])]

        fMln = pf.readline().split(comm,1)[0].split(sep)
        fM = [float(fMln[0]), float(fMln[1])]

    rhobd = st.uniform(loc=rhob[0], scale=rhob[1]-rhob[0])
    Vbd = st.uniform(loc=Vb[0], scale=Vb[1]-Vb[0])
    rhold = st.uniform(loc=rhol[0], scale=rhol[1]-rhol[0])
    Vld = st.uniform(loc=Vl[0], scale=Vl[1]-Vl[0])
    hd = st.uniform(loc=h[0], scale=h[1]-h[0])
    dhd = st.uniform(loc=dh[0], scale=dh[1]-dh[0])
    fMd = st.uniform(loc=fM[0], scale=fM[1]-fM[0])

    priors = [rhobd, Vbd, rhold, Vld, hd, dhd, fMd]
    return priors

def exponential_covariance(t, alpha, sigma):
    '''
    alpha: time for signal to decay to sigma/e
    '''
    dists = np.abs(t[:,np.newaxis]-t.T)
    Sigma = sigma**2 * np.exp(-dists/alpha)
    return Sigma

def sample_data(N,dmean,dcov):
    '''
    This function takes mean and standard deviation vectors and generates N samples

    NOTE: dstd will eventually be replaces with a covariance matrix

    Inputs:
        N
        dmean
        dstd

    Outputs:
        D_N
    '''
    return multivariate_normal.rvs(dmean, dcov, N)

        
def train_reg(dtrain, vtrain):
    '''
    This function trains a random forest 
    '''
    return 

def add_noise(D, cov):
    '''
    This function takes the data matrix D and adds noise according to the error covariance in cov
    '''
    noise = multivariate_normal.rvs(np.zeros(D.shape[1]), cov, D.shape[0])
    return D+noise


def train_inverse(D, inputs, model="RandomForest"):
    if model is "RandomForest":
        return train_RF(D, inputs)
    elif model is "CNN":
        return train_CNN(D, inputs)
    else:
        return

def train_CNN(D, inputs):

    nparams = inputs.shape[1]
    ndata = D.shape[1]

    xtrain = D.reshape(D.shape[0], D.shape[1], 1)
    ytrain = inputs

    model = Sequential()
    model.add(Conv1D(32, 3, activation="relu", input_shape=(ndata,1))) #32 filters, kernel size 2
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(nparams))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)

    def MLinv(data):
        x = data.reshape(data.shape[0], data.shape[1], 1)
        return model.predict(x)

    return MLinv


def train_RF(D, inputs):
    '''
    This function trains Random Forests to predict each model parameter in inputs.

    Inputs:
        D: [SxD] matrix of training data
        inputs: [SxP] matrix of training parameters

    Outputs:
        models: [Px1] list of random forest regressors
    '''
    nparams = inputs.shape[1]
    models = [None]*nparams
    for i in range(nparams):
        print('Training '+str(i+1)+' of '+str(nparams))
        models[i] = RandomForestRegressor().fit(D, inputs[:,i])

    def MLinv(D):
        params = np.zeros((D.shape[0], len(models)))
        for i in range(len(models)):
            params[:,i] = models[i].predict(D)
        return params

    return MLinv


def plot_trace(D,t, xlim=(-.2,.2)):
    '''

    '''
    plt.plot(D,t)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((xlim[0],xlim[1]))
    plt.xlabel('Amplitude')
    plt.ylabel('Two-Way Travel Time (s)')
    return

def gen_multiple_models(inputs, zmax=1000):
    '''

    '''
    
    z = np.linspace(0,zmax,100)
    impb = inputs[:,0]*inputs[:,1]
    impl = inputs[:,2]*inputs[:,3]
    itop = np.argmin(np.abs(z[:,np.newaxis]-inputs[:,4]),axis=0)
    ibot = np.argmin(np.abs(z[:,np.newaxis]-(inputs[:,4]+inputs[:,5])),axis=0)
    imp = np.ones((inputs.shape[0],z.shape[0])) * impb[:,np.newaxis]
    for i in range(inputs.shape[0]):
        imp[i,itop[i]:ibot[i]] = impl[i]

    return imp, z

def gen_single_model(inputs, zmax=1000):
    '''

    '''
    z = np.linspace(0,zmax,100)
    impb = inputs[0]*inputs[1]
    impl = inputs[2]*inputs[3]
    itop = np.argmin(np.abs(z-inputs[4]))
    ibot = np.argmin(np.abs(z-inputs[4]-inputs[5]))
    imp = np.ones(z.shape) * impb
    imp[itop:ibot] = impl

    return imp, z

def gen_model(inputs, zmax=1000):
    '''

    '''
    if inputs.ndim is 1:
        return gen_single_model(inputs, zmax=zmax)
    else:
        return gen_multiple_models(inputs, zmax=zmax)


def plot_model(inputs, zmax=1000, xlim=None):
    imp, z = gen_model(inputs,zmax=zmax)
    plt.plot(imp/10**6,z)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if xlim is not None:
        plt.xlim((xlim[0],xlim[1]))
    plt.xlabel('Impedance')
    plt.ylabel('Depth (m)')
    return

def plot_2_models(inputs1, inputs2, zmax=1000, xlim=None, lbl1='Model 1', lbl2='Model 2'):
    '''

    '''
    plt.subplots(nrows=1, ncols=2)
    plt.subplot(121)
    plot_model(inputs1,xlim=(3,7.5))
    plt.title(lbl1)
    plt.subplot(122)
    plot_model(inputs2,xlim=(3,7.5))
    plt.title(lbl2)
    plt.tight_layout()
    return

def plot_2_traces(inputs1, inputs2, t, xlim=None, lbl1='Model 1', lbl2='Model 2'):
    '''

    '''
    plt.subplots(nrows=1, ncols=2)
    plt.subplot(121)
    plot_trace(inputs1,t)
    plt.title(lbl1)
    plt.subplot(122)
    plot_trace(inputs2, t)
    plt.title(lbl2)
    plt.tight_layout()
    return

def plot_trace_model(D,t,inputs):
    '''

    '''
    plt.subplots(nrows=1, ncols=2)
    plt.subplot(121)
    plot_trace(D,t)
    plt.title('Seismic Trace')
    plt.subplot(122)
    plot_model(inputs,xlim=(3,7.5))
    plt.title('Impedance Profile')
    plt.tight_layout()
    return

def explore_prior(ind, Dprior, t, inputs):
    '''

    '''
    plot_trace_model(Dprior[ind,:],t,inputs[ind,:])
    return

def explore_traces(ind, D, t):
    '''

    '''
    plot_trace(D[ind,:], t)
    return

def explore_posterior_models(ind, posterior, truth):
    '''

    '''
    plot_2_models(posterior[ind,:], truth, lbl1='Posterior', lbl2='Truth')
    return

def explore_posterior_traces(ind, Dposterior, Dobs, t):
    '''

    '''
    plot_2_traces(Dposterior[ind,:], Dobs[ind,:], t, lbl1='Posterior Prediction', lbl2='Observed')
    return
    

def plot_predictions(p_pred_prior, p_true_prior, ntrain=None, p_pred_obs=None, p_true_obs=None, varnames=None, units=None, figsize=(14,12)):
    '''
    This function plots a scatter plot of true vs predicted paramters
    '''
    nparams = p_pred_prior.shape[1]
    nrows = np.floor(nparams/2+1).astype(np.int)
    plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    
    for i in range(nparams):
        pltnum = nrows*100+20+1+i
        plt.subplot(pltnum)

        pmin = np.min(p_pred_prior[:,i])
        pmax = np.max(p_pred_prior[:,i])

        pmin_true = np.min(p_true_prior[:,i])
        pmax_true = np.max(p_true_prior[:,i])
        aa = (pmax_true-pmin_true) * 0.1

        if ntrain is not None:
            plt.scatter(p_pred_prior[ntrain+1:,i], p_true_prior[ntrain+1:,i], color=(.25,.25,1), marker='s', edgecolor='k', label='Test')
            plt.scatter(p_pred_prior[:ntrain,i], p_true_prior[:ntrain,i], color=(1,.5,.5), marker='o', edgecolor='k', label='Train')
        else:
            plt.scatter(p_pred_prior[:,i], p_true_prior[:,i], color=(.25,.25,1), marker='s', edgecolor='k', label='Test')
        
        if p_pred_obs is not None:
            p50 = np.percentile(p_pred_obs[:,i],50)
            p2 = np.percentile(p_pred_obs[:,i],2.5)
            p97 = np.percentile(p_pred_obs[:,i],97.5)

            plt.plot([p50, p50], [pmin_true-aa, pmax_true+aa],'-k',linewidth=3, label='Median')
            plt.plot([p2, p2], [pmin_true-aa, pmax_true+aa],':k',linewidth=3,label='95% Confidence Interval')
            plt.plot([p97, p97], [pmin_true-aa, pmax_true+aa],':k',linewidth=3)

        plt.plot([pmin,pmax], [pmin,pmax],'-c',linewidth=2,label='1:1')
        xl = plt.gca().get_xlim()

        if p_true_obs is not None:
            plt.plot(xl,[p_true_obs[i],p_true_obs[i]], '-r', linewidth=3, label='True')

        if varnames is not None:
            if units is not None:
                plt.title(varnames[i] + ' ('+units[i]+')',fontsize=16)
            else:
                plt.title(varnames[i], fontsize=16)
        
        plt.ylim((pmin_true-aa,pmax_true+aa))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.subplot(nrows*100+20+1)
    plt.legend()
    plt.subplot(nrows*100+20+nrows*2-1)
    plt.xlabel('Predicted',fontsize=16)
    plt.ylabel('True',fontsize=16)

    plt.tight_layout()
    return
    
def plot_predictions2(p_pred_prior, p_true_prior, ntrain=None, p_pred_obs=None, p_true_obs=None, varnames=None, units=None, figsize=(14,12), markersize=1):
    '''
    This function plots a scatter plot of true vs predicted paramters
    '''
    nparams = p_pred_prior.shape[1]
    nrows = np.floor(nparams/2+1).astype(np.int)
    plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    
    for i in range(nparams):
        pltnum = nrows*100+20+1+i
        plt.subplot(pltnum)

        pmin = np.min(p_pred_prior[:,i])
        pmax = np.max(p_pred_prior[:,i])

        pmin_true = np.min(p_true_prior[:,i])
        pmax_true = np.max(p_true_prior[:,i])
        aa = (pmax_true-pmin_true) * 0.1

        if ntrain is not None:
            plt.scatter(p_pred_prior[ntrain+1:,i], p_true_prior[ntrain+1:,i], markersize, color=(.25,.25,1), marker='.', label='Test')
            plt.scatter(p_pred_prior[:ntrain,i], p_true_prior[:ntrain,i], markersize, color=(1,.5,.5), marker='.', label='Train')
        else:
            plt.scatter(p_pred_prior[:,i], p_true_prior[:,i], color=(.25,.25,1), marker='s', edgecolor='k', label='Test')
        
        if p_pred_obs is not None:
            p50 = np.percentile(p_pred_obs[:,i],50)
            p2 = np.percentile(p_pred_obs[:,i],2.5)
            p97 = np.percentile(p_pred_obs[:,i],97.5)

            plt.plot([p50, p50], [pmin_true-aa, pmax_true+aa],'-k',linewidth=3, label='Median')
            plt.plot([p2, p2], [pmin_true-aa, pmax_true+aa],':k',linewidth=3,label='95% Confidence Interval')
            plt.plot([p97, p97], [pmin_true-aa, pmax_true+aa],':k',linewidth=3)

        plt.plot([pmin,pmax], [pmin,pmax],'-k',linewidth=3,label='1:1')
        xl = plt.gca().get_xlim()

        if p_true_obs is not None:
            plt.plot(xl,[p_true_obs[i],p_true_obs[i]], '-r', linewidth=3, label='True')

        if varnames is not None:
            if units is not None:
                plt.title(varnames[i] + ' ('+units[i]+')',fontsize=16)
            else:
                plt.title(varnames[i], fontsize=16)
        
        plt.ylim((pmin_true-aa,pmax_true+aa))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.subplot(nrows*100+20+1)
    plt.legend()
    plt.subplot(nrows*100+20+nrows*2-1)
    plt.xlabel('Predicted',fontsize=16)
    plt.ylabel('True',fontsize=16)

    plt.tight_layout()
    return
def plot_posteriors(posterior, priors=None, truths=None, varnames=None, units=None, figsize=(14,12)):
    nparams = posterior.shape[1]
    nrows = np.floor(nparams/2+1).astype(np.int)
    plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    
    if priors is not None:
        pds = load_priors(priors)
    
    for i in range(nparams):
        pltnum = nrows*100+20+1+i
        plt.subplot(pltnum)
        
        plt.hist(posterior[:,i],density=True,facecolor=(.25,.25,1),edgecolor='k',label='Posterior')
        
        if priors is not None:
            aa = (pds[i].ppf(0.999) - pds[i].ppf(0.001)) * .1
            xs = np.linspace(pds[i].ppf(0.001)-aa, pds[i].ppf(0.999)+aa, 1000)
            probs = pds[i].pdf(xs)
            plt.plot(xs,probs,'-k',linewidth=3,label='Prior')
            
        yl = plt.gca().get_ylim()
        
        if truths is not None:
            plt.plot([truths[i],truths[i]], yl, '-r', linewidth=3, )
        
        if varnames is not None:
            if units is not None:
                plt.xlabel(varnames[i] + ' ('+units[i]+')',fontsize=16)
            else:
                plt.xlabel(varnames[i], fontsize=16)  
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
    plt.subplot(nrows*100+20+1)
    plt.legend()
    plt.subplot(nrows*100+20+nparams)
    plt.ylabel('Probability Density',fontsize=16)
    
    plt.tight_layout()
        
    return

def plot_model_ensemble(prior, posterior, truth):
    imp, z = gen_model(posterior)
    impt, zt = gen_model(truth)
    impp, zp = gen_model(prior)

    plt.plot(impp[0,:]/10**6,zp,color=(.75,.75,.75),label='Prior')
    for i in range(1,impp.shape[0]):
        plt.plot(impp[i,:]/10**6,zp,color=(.75,.75,.75))

    plt.plot(imp[0,:]/10**6,z,color=(.5,.5,.5),label='Posterior')
    for i in range(1,imp.shape[0]):
        plt.plot(imp[i,:]/10**6,z,color=(.5,.5,.5))
    
    plt.plot(impt/10**6,zt, color='k',label='Truth')
    plt.gca().invert_yaxis()
    plt.xlabel('Impedance (/10^6)')
    plt.ylabel('Depth (z)')
    plt.legend()
    return
           
#def plot_importances():


