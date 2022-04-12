import numpy as np
import pandas as pd
from pymc import Uniform, Normal, deterministic, MCMC, Matplot, AdaptiveMetropolis
from Functions import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

# species
idx = 0
species_dict1 = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
codes = list(species_dict1.keys())
code = codes[idx]
species_dict2 = {'Aru':[31, 48], 'Bpa':[56, 144], 'Pgr':[61, 122], 'Pst':[63, 142], 'Qru':[51, 88]}
species = species_dict1[code]
Vcmax, Jmax = species_dict2[species]
species_code = species+'_'+str(code)

# read csv
df = pd.read_csv('UMB_daily_average_Gil_v2.csv')

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
#df = df.iloc[54:86, ] # with ps < -0.1
df[species_code] = df[species_code].replace({0: np.nan})
df = df.dropna()
#df = df.drop(df.index[list(range(76, 79))])
T = df['T']
I = df['I']
D = df['D']
ps = df['ps']
vn = df[species_code]
vn_max = vn.max()
vn = vn/vn_max

''' Priors '''
alpha_log10 = Uniform('alpha_log10', lower=-5, upper=0, value=-2.2)
c = Uniform('c', lower=5, upper=30, value=10)
g1 = Uniform('g1', lower=0.01, upper=1, value=0.5)
kxmax_log10 = Uniform('kxmax_log10', lower=0, upper=3, value=2)
p50 = Uniform('p50', lower=-5, upper=-0.1, value=-1)

''' deterministic model '''
@deterministic
def muf(alpha_log10=alpha_log10, c=c, g1=g1, kxmax_log10=kxmax_log10, p50=p50,
        ca=400, Kc=460, q=0.3, R=8.314, Vcmax=Vcmax, Jmax=Jmax, z1=0.9, z2=0.9999, a=1.6, rho=997000, L=1):
    alpha, kxmax = 10**alpha_log10, 10**kxmax_log10
    sapflow_modeled = []
    for i in range(len(vn)):
        # Environmental conditions
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
                sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*18/1000000/1000*rho/alpha/vn_max)
            else:
                #print(i, ' ', c, ' ', p50, ' ', Ti)
                if abs(px1) < abs(px2):
                    sapflow_modeled.append(1)
                else:
                    sapflow_modeled.append(0)
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
    return sapflow_modeled

'''data likelihoods'''
rel_std = 0.1
precision = 1/(rel_std*vn.mean())**2
Y_obs = Normal('Y_obs', mu=muf, tau=precision, value=vn, observed=True)

''' posterior sampling '''
M = MCMC([alpha_log10, c, g1, kxmax_log10, p50])
M.use_step_method(AdaptiveMetropolis, [alpha_log10, c, g1, kxmax_log10, p50])
M.sample(iter=1000000, burn=500000, thin=40)
#M.sample(iter=500000, burn=250000, thin=20)

# Save trace
ensure_dir('{}'.format(species_code))
traces = {'alpha_log10':M.trace('alpha_log10')[:], 'c':M.trace('c')[:], 'g1':M.trace('g1')[:],
          'kxmax_log10':M.trace('kxmax_log10')[:], 'p50':M.trace('p50')[:]}
pickle_out = open('{}.pickle'.format(species_code), 'wb')
pickle.dump(traces, pickle_out)
pickle_out.close()

# Trace
Matplot.plot(M)
