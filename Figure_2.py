import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_codes.remove('Pst_1')
species_codes.remove('Pst_19')
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                     'Bpa': ['Paper birch', 'Betula papyrifera'],
                     'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                     'Pst': ['White pine', 'Pinus strobus']}

df = None
for species_code in species_codes:
    if not os.path.exists('Figure_1&2&4/{}.pickle'\
                          .format(species_code)):
        continue
    ts = pickle.load(open('Figure_1&2&4/{}.pickle'\
                          .format(species_code), 'rb'))['c']
    temp = pd.DataFrame(ts, columns=['c'])
    temp['species_code'] = species_code
    temp['Species'] = species_name_dict[species_code[:3]][0]
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])
df = df.sort_values(by=['species_code'])
#df = df[~df['species_code'].isin(['Bpa_31', 'Pgr_27', 'Pst_19'])]

# figure
plt.figure(figsize=(20, 10))
sns.set(font_scale=2.5)
sns.set_style('white')
g = sns.violinplot(x='species_code', y='c', hue='Species', data=df,\
                   cut=0, dodge=False)
g.set(xticklabels=[])
g.set(xlabel=None)
g.tick_params(bottom=False)
g.set_ylabel(ylabel='$c$')
g.set(ylim=(0, 40))
g.legend(loc='upper center', title_fontsize=30, ncol=4,\
         bbox_to_anchor=(0.5, 1.12))
fig = g.get_figure()
# fig.savefig('../../Figures/c.png', bbox_inches='tight')
# print(df.groupby('species_code')['c'].agg(['mean', 'std'])\
#       .apply(lambda x: x['std']/x['mean'], axis=1).mean())