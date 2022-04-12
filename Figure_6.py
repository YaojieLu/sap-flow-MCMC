import numpy as np
import pandas as pd
import os
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import sklearn as sck
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.graphics.gofplots as sm1
from matplotlib.lines import Line2D

# read csv
dff = pd.read_csv('./MCMC/UMB_daily_average_Gil_v2.csv')

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                  7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                  17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                  26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                  32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
# species_codes = { 3: 'Pgr', 4: 'Pgr', 22: 'Pgr',
#                  27: 'Pgr', 34: 'Pgr'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
for species_code in species_codes:
    if not os.path.exists('./Figure_6/ensemble_px_UMB_Gil_v2_{}.csv'\
                          .format(species_code)):
        species_codes.remove(species_code)
species_codes.sort()
species_codes.remove('Pst_1')
species_codes.remove('Pst_19')


# extract data
dff = dff[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', 'date',species_code]]
dff['ps'] = dff[['ps15', 'ps30', 'ps60']].mean(1)
#df = df.iloc[54:86, ] # with ps < -0.1
dff[species_code] = dff[species_code].replace({0: np.nan})
#dff = dff.dropna()
#df = df.drop(df.index[list(range(76, 79))])
T = dff['T']
I = dff['I']
D = np.sqrt(dff['D']*101)*-1
ps = dff['ps']
vn = dff[species_code]
vn_max = vn.max()
vn = vn/vn_max

# figure
species_abbrs = ['Aru', 'Bpa', 'Pgr', 'Pst']
species_names = ['Red maple', 'Paper birch', 'Bigtooth aspen',\
                 'Eastern white pine']
species_names = dict(zip(species_abbrs, species_names))
colors = sns.color_palette()[:4]
colors = dict(zip(species_abbrs, colors))
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex=True,\
                        sharey=True)
# store slopes and intercepts
dim = 1
fit_df = pd.DataFrame(data = np.zeros(shape = (len(species_codes),6)),\
                      columns = ['species','slope1','slope2','slope3','intercept','R2'])
slps = np.zeros(shape = (len(species_codes),dim))
itcpts = np.zeros(len(species_codes))
i = 0
for row in axs:
    for col in row:
        if i < len(species_codes):
            species_code = species_codes[i]
            
            df = pd.read_csv('./Figure_S3/ensemble_px_UMB_Gil_v2_{}.csv'\
                             .format(species_code))
            columns = ['qt=0.05', 'qt=0.5', 'qt=0.95']
            # Linear regression
            X1 = dff['ps'].to_numpy()
            X2 = 1/np.sqrt(dff['D'].to_numpy()*101)
            #X2 = (dff['D'].to_numpy()*101)
            X3 = X1*X2
            Y = df['qt=0.5'].to_numpy()
            rem = np.isnan(X1) | np.isnan(X2) | np.isnan(Y)
            Xplot = X1[~rem].reshape(-1,1);
            Xd = X1[~rem].reshape(-1,1)
            Xd2 = np.append(Xd,X2[~rem].reshape(-1,1),1)
            Xd3 = np.append(Xd2,X3[~rem].reshape(-1,1),1)
            Xd = sm.add_constant(Xd)
            Xd2 = sm.add_constant(Xd2)
            Xd3 = sm.add_constant(Xd3)
            Yd = Y[~rem].reshape(-1,1)
            
            if dim == 1:
            
                # Fit OLS to soil water potential
                lr = sm.OLS(Yd,Xd)
#                lr = sm.RLM(Yd,Xd)
                results = lr.fit()
#                print(results.summary2())
            elif dim==2:
                # Fit OLS to soil water potential and VPD
                lr = sm.OLS(Yd,Xd2)
                results = lr.fit()
                fit_df['slope2'].iloc[i] = results.params[2]
#                print(results2.summary())
            else:
                # Fit OLS to soil water potential and VPD and an interaction term
                lr = sm.OLS(Yd,Xd3)
                results = lr.fit()
                fit_df['slope2'].iloc[i] = results.params[2]
                fit_df['slope3'].iloc[i] = results.params[3]
                
            fit_df['species'].iloc[i] = species_names[species_codes[i][0:3]]
            fit_df['slope1'].iloc[i] = results.params[1]
            fit_df['intercept'].iloc[i] = results.params[0]
            fit_df['R2'].iloc[i] = results.rsquared_adj


            Y_pred = results.fittedvalues
            Res = results.resid
            
            col.scatter(dff['ps'], df['qt=0.5'], color=colors[species_code[:3]])
            #col.scatter(Xplot, Y_pred, color='black',marker = "x")
            col.plot(Xplot, Y_pred, color='black',linestyle='-')
            
            slps = fit_df['slope1'].iloc[i]
            slps2 = fit_df['slope2'].iloc[i]
            itcpts = fit_df['intercept'].iloc[i]
            R2 = fit_df['R2'].iloc[i]
            col.text(-1.3,-0.3,species_codes[i],fontsize = 15)
            col.text(-1.3,-0.5,r'$\sigma_s$ = ' + str(slps.round(decimals = 3).item()),fontsize = 15)
            col.text(-1.3,-0.7,r'$\Lambda$ = ' + str(itcpts.round(decimals = 3).item()),fontsize = 15)
            col.text(-1.3,-0.9,r'$R^2$ = ' + str(R2.round(decimals = 3).item()),fontsize = 15)

            i += 1
        else:
            i += 1
            date = pd.to_datetime(df.date).apply(lambda x: x.strftime('%m/%d'))
            #col.plot(dff['ps'], df['qt=0.5'], lw=0)
        col.xaxis.set_major_locator(ticker.MultipleLocator(1))
        col.yaxis.set_major_locator(ticker.MultipleLocator(1))
        #col.xaxis.set_tick_params(rotation=40)
        col.tick_params(labelsize=30)
plt.subplots_adjust(hspace=0.2, wspace=0.1)
fig.text(0.08, 0.5, '$\psi_x$ (MPa)',\
         va='center', rotation='vertical', fontsize=30)
fig.text(0.5, 0.08, '$\psi_s$ (MPa)',\
         va='center', fontsize=30)
custom_lines = [Line2D([0], [0], lw=4, color='black', label='Fit')]+\
               [Line2D([0], [0], lw=4,\
                       color=colors[s], label=species_names[s])\
                for s in species_abbrs]
fig.legend(handles=custom_lines, prop={'size': 30}, loc='upper center',\
          ncol=len(species_abbrs)+1, frameon=False)
fig.subplots_adjust(top=0.9)
fig.savefig('AllFits.svg', bbox_inches='tight')


# =============================================================================
# # Plots of the slope and intercept fits for species
# =============================================================================
sns.set(font_scale=1)
sns.set_style('white')

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1, 1]},\
                        figsize=(10, 5), sharey=False)

sns.boxplot(x='species', y='slope1', data=fit_df,ax = axs[0],fliersize=5)
# sns.scatterplot(x='species', y='slope1', data=fit_df, s=25,\
#                 legend=False,ax=axs[0],style = 'species',hue = 'species')
axs[0].tick_params(bottom=True,left=True)
axs[0].set(xlabel='')
axs[0].set(ylabel='Isohydry Index, $\sigma$ [-]')
axs[0].xaxis.set_tick_params(rotation=40)
    
mrk = ["o","X","s","P"]
sites = []
jj = 0
for j in species_abbrs:
        td = fit_df['species']==species_names[j]
        tempprms = fit_df[td]
        sel = np.where(td)[0].tolist()
        tempnames = [species_codes[i] for i in sel]
        selprm = np.argmin(np.abs(tempprms['slope1'] - tempprms['slope1'].median()))
        #selprm = np.where(selprm)[0].tolist()
        sites.append(tempnames[selprm])
        df = pd.read_csv('./Figure_S3/ensemble_px_UMB_Gil_v2_{}.csv'\
                             .format(tempnames[selprm]))  
        sns.scatterplot(x=dff['ps'], y=df['qt=0.5'],ax = axs[1],marker=mrk[jj],s = 25)
        xLim = np.array([-1.5,0])
        yLim = tempprms['slope1'].iloc[selprm]*xLim + tempprms['intercept'].iloc[selprm]
        sns.lineplot(xLim, yLim,linestyle='-',ax = axs[1])
        jj = jj + 1
 
axs[1].tick_params(bottom=True,left=True)
axs[1].set(xlabel='$\psi_s$ (MPa)')
axs[1].set(ylabel='$\psi_x$ (MPa)')
axs[1].legend(sites + sites)
plt.subplots_adjust(wspace=0.3)
fig.savefig('test.svg', bbox_inches='tight')


# =============================================================================
# # Plots of the slope and intercept fits for species
# =============================================================================
sns.set(font_scale=1)
sns.set_style('white')

fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1, 1, 1]},\
                        figsize=(15, 5), sharey=False)

sns.scatterplot(x='species', y='slope1', hue='species', data=fit_df, s=99,\
                legend=False,ax=axs[0])

axs[0].tick_params(bottom=True,left=True)
axs[0].set(xlabel='')
axs[0].set(ylabel='Slope, $\sigma_s$ [-]')
axs[0].xaxis.set_tick_params(rotation=40)

 
sns.scatterplot(x='species', y='intercept', hue='species', data=fit_df, s=99,\
                legend=False,ax=axs[1])
axs[1].tick_params(bottom=True,left=True)
axs[1].set(xlabel='')
axs[1].set(ylabel='Intercept, $\Lambda$ [MPa]')
axs[1].xaxis.set_tick_params(rotation=40)

sns.scatterplot(x='species', y='R2', hue='species', data=fit_df, s=99,\
                legend=False,ax=axs[2])
axs[2].tick_params(bottom=True,left=True)
axs[2].set(xlabel='')
axs[2].set(ylabel='$R^2$')
axs[2].xaxis.set_tick_params(rotation=40)

plt.subplots_adjust(wspace=0.3)


# =============================================================================
# # ANOVA analysis of the plot parameters
# =============================================================================
from statsmodels.formula.api import ols
var2test = 'slope1'
mdl = ols(var2test + ' ~ C(species)',data=fit_df).fit()
anova_table = sm.stats.anova_lm(mdl, typ=2)
anova_table
tuk = sm.stats.multicomp.pairwise_tukeyhsd(fit_df[var2test],fit_df['species'],alpha = 0.05)
#tuk = sm.stats.multicomp.pairwise_tukeyhsd(fit_df['slope1'].to_numpy().reshape(-1,1))
print(tuk)


# =============================================================================
# # Time series plots for soil and leaf water potential as well as VPD
# =============================================================================
#fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex=True,\
#                        sharey=True)
#i = 0
#for row in axs:
#    for col in row:
#        if i < len(species_codes):
#            species_code = species_codes[i]
#            i += 1
#            df = pd.read_csv('./Figure_S3/ensemble_px_UMB_Gil_v2_{}.csv'\
#                             .format(species_code))
#            columns = ['qt=0.05', 'qt=0.5', 'qt=0.95']
#            #df = df.dropna()
##            col.plot(df['date'], dff['ps15'], color='lightgray')
##            col.plot(df['date'], dff['ps30'], color='silver')
##            col.plot(df['date'], dff['ps60'], color='darkgrey')
#            col.plot(df['date'], D, color='lightgray')
#            col.plot(df['date'], df['qt=0.5'], color=colors[species_code[:3]])
#            col.plot(df['date'], dff['ps'], color='black')
#
##            col.fill_between(df['date'], df['qt=0.05'], df['qt=0.95'],\
##                             color='b', alpha=0.2)
#        else:
#            i += 1
#            date = pd.to_datetime(df.date).apply(lambda x: x.strftime('%m/%d'))
#            col.plot(date, df['qt=0.5'], lw=0)
#        col.xaxis.set_major_locator(ticker.MultipleLocator(50))
#        #col.xaxis.set_tick_params(rotation=40)
#        col.tick_params(labelsize=30)
#plt.subplots_adjust(hspace=0.2, wspace=0.1)
#fig.text(0.08, 0.5, '$\psi_x$ (MPa) of -VPD (kPa)',\
#         va='center', rotation='vertical', fontsize=30)
#fig.text(0.5, 0.08, 'Date',\
#         va='center', fontsize=30)
#custom_lines = [Line2D([0], [0], lw=4, color='lightgray', label='$-VPD$')]+\
#                [Line2D([0], [0], lw=4, color='black', label='$\psi_s$')]+\
#               [Line2D([0], [0], lw=4,\
#                       color=colors[s], label=species_names[s])\
#                for s in species_abbrs]
#fig.legend(handles=custom_lines, prop={'size': 30}, loc='upper center',\
#          ncol=len(species_abbrs)+1, frameon=False)
#fig.subplots_adjust(top=0.9)
##fig.savefig('../../Figures/ensemble_px_dynamics.png', bbox_inches='tight')
