import pickle
from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# load MCMC output
baseline = pickle.load(open('Figure_9/{}.pickle'\
                            .format('baseline'), 'rb'))
p50 = pickle.load(open('Figure_9/{}.pickle'\
                       .format('prior_p50'), 'rb'))['p50']
c = pickle.load(open('Figure_9/{}.pickle'\
                     .format('prior_c'), 'rb'))['c']
x_p50 = np.linspace(-5, -0.1, 1000)
x_c = np.linspace(5, 30, 1000)
def get_truncated_normal(mean=-2.55, sd=1, low=-5, upp=-0.1):
    return truncnorm(
        (low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)
rv_p50 = get_truncated_normal()
rv_c = get_truncated_normal(mean=17.5, sd=5, low=5, upp=30)
true_p50 = -2.5
true_c = 13

# figure
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
# prior
axs[0].plot([-5, -0.1], [1/4.9, 1/4.9], color='red', linewidth=4)
axs[0].plot(x_p50, rv_p50.pdf(x_p50), color='blue', linewidth=4)
axs[0].set_xticks([-5, -0.1])
axs[0].set_xticklabels(['min', 'max'])
axs[0].tick_params(labelsize=45)
axs[0].set_xlabel('Prior distribution', fontsize=45)
axs[0].axes.get_yaxis().set_ticks([])
# p50
axs[1].hist(baseline['p50'], bins=30, color='red', histtype='step',\
            density=True, linewidth=4)
axs[1].hist(p50, bins=30, color='blue', histtype='step', density=True,\
            linewidth=4)
axs[1].axvline(x=true_p50, linewidth=8, color='black')
axs[1].tick_params(labelsize=45)
axs[1].set_xlabel('P50 (MPa)'
                  '\n'
                  'Posterior distribution', fontsize=45)
axs[1].axes.get_yaxis().set_ticks([])
# c
axs[2].hist(baseline['c'], bins=30, color='red', histtype='step',\
            density=True, linewidth=4)
axs[2].hist(c, bins=30, color='blue', histtype='step', density=True,\
            linewidth=4)
axs[2].axvline(x=true_c, linewidth=8, color='black')
axs[2].tick_params(labelsize=45)
axs[2].set_xlabel(r'$c$'
                  '\n'
                  'Posterior distribution', fontsize=45)
axs[2].axes.get_yaxis().set_ticks([])
custom_lines = [Line2D([0], [0], color='red', label='Uniform prior', lw=4),
                Line2D([0], [0], color='blue', lw=4,\
                       label='Truncated normal prior'),
                Line2D([0], [0], color='black', label='True value', lw=8)]
fig.legend(loc='upper center', handles=custom_lines, ncol=3,\
           prop={'size': 40}, bbox_to_anchor=(0.425, 1.12))
fig.subplots_adjust(wspace=0.1)
#fig.savefig('../../../Figures/Prior.png', bbox_inches='tight')
