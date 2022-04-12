import numpy as np
import pandas as pd

# read file and preprocessing
year = 2015
df = pd.read_csv('/UMB_rawdata.csv')
df = df[df['year']==year]
df = df.iloc[9:-39, :]
df = df.replace(r'^\s+$', np.nan, regex=True)
day_len = np.where(df['ppfd_in'] >= 10, 1, np.where(df['ppfd_in'] < 10, 0,\
                                                    np.nan)).reshape(-1, 48)

# column names
ps_list = ['ps15', 'ps30', 'ps60']
swc_list = ['SWC_1_{}_1'.format(i+3) for i in range(3)]
species_dict = {'Aru_7': [123.64, 13.8], 'Aru_9': [123.64, 13.8],
                'Aru_11': [123.64, 13.8], 'Aru_21': [123.64, 13.8],
                'Aru_24': [123.64, 13.8], 'Aru_26': [123.64, 13.8],
                'Aru_28': [123.64, 13.8], 'Aru_29': [123.64, 13.8],
                'Bpa_31': [198.57, 18.1], 'Bpa_32': [198.57, 18.1],
                'Bpa_35': [198.57, 18.1], 'Bpa_38': [198.57, 18.1],
                'Bpa_41': [198.57, 18.1],
                'Pgr_3': [239.47, 21.6], 'Pgr_4': [239.47, 21.6],
                'Pgr_22': [239.47, 21.6], 'Pgr_27': [239.47, 21.6],
                'Pgr_34': [239.47, 21.6],
                'Pst_1': [114.5, 15], 'Pst_2': [114.5, 15],
                'Pst_6': [114.5, 15], 'Pst_12': [114.5, 15],
                'Pst_14': [114.5, 15], 'Pst_17': [114.5, 15],
                'Pst_19': [114.5, 15]}

env_list = ['T', 'I', 'D']
date = ['year', 'month', 'day']

# the van Genuchten model
def psf(s, ss=0.37, sr=0.04, alpha=-5.2, n=1.68):
    return 1/alpha*(((ss-sr)/(s/100-sr))**(n/(n-1))-1)**(1/n)*0.009804139432

# daily
def meanf(x):
    xre = x.values.reshape(-1, 48)
    xre2 = xre*day_len
    xre2[xre2 == 0] = np.nan
    filter = np.where(np.count_nonzero(xre2*day_len>0, axis=1)<8, np.nan, 1)
    res = np.nanmean(xre2, axis=1)*filter
    #res = pd.Series(res).interpolate()
    return res
def sumf(x):
    xre = x.values.reshape(-1, 48)
    xre2 = xre*day_len
    xre2[xre2 == 0] = np.nan
    res = np.nansum(xre2, axis=1)
    #res = pd.Series(res).interpolate()
    return res
def maxf(x):
    xre = x.values.reshape(-1, 48)
    xre2 = xre*day_len
    xre2[xre2 == 0] = np.nan
    res = np.nanmax(xre2, axis=1)
    #res = pd.Series(res).interpolate()
    return res
def minf(x):
    xre = x.values.reshape(-1, 48)
    xre2 = xre*day_len
    xre2[xre2 == 0] = np.nan
    res = np.nanmin(xre2, axis=1)
    #res = pd.Series(res).interpolate()
    return res
def medf(x):
    xre = x.values.reshape(-1, 48)
    res = np.nanmedian(xre, axis=1)
    return res

# transformation
for ps, swc in zip(ps_list, swc_list):
    df[ps] = df.apply(lambda row: psf(row[swc]), axis=1)
df['D'] = df['D']/101.325
df_da = df[ps_list].apply(minf)
df_da['D'] = df[['D']].apply(maxf)
df_da[date] = df[date].apply(medf)
df_da[list(species_dict.keys())] = df[list(species_dict.keys())].apply(meanf)
df_da[['T', 'I']] = df[['T', 'I']].apply(meanf)
df_da['day_len'] = np.nansum(day_len, axis=1)
# for s in species_dict.keys():
#     crown_area = 0.25*np.pi*(2.67*np.log(species_dict[s][1])-1.9)**2
#     df_da[s] = df_da[s]*species_dict[s][0]/10000/crown_area*60*30/1000
df_da['date'] = pd.to_datetime((df_da['year']*10000+df_da['month']*100+\
                                df_da['day']).apply(str), format='%Y%m%d')
df_da = df_da.drop(['year', 'month', 'day'], axis=1)

# output
df_da.replace([np.inf, -np.inf], np.nan, inplace=True)
df_da.to_csv('../MCMC/UMB_daily_average_Gil_v2.csv', index=False)
