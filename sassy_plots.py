import pandas as pd
import json, argparse, os
import pylab as plt
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
        'axes.labelsize' : 16,
        'font.size' : 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'text.usetex': False,
        'figure.figsize': [10, 8]
        }
matplotlib.rcParams.update(params)


def histedges_equalN(x, nbin):
    """
    Generates 1D histogram with equal
    number of examples in each bin
    :param x: input data
    :param nbin: number of bins
    :return: bin edges
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def snr_scatter(param, ax, fig, df):
    """
    Scatter plot of Recovered S/N vs Injected
    S/N with colorscale as FRB property/parameter
    :param param: parameter to use for colorscale
    :param ax: axis object
    :param fig: figure object
    :param df: dataframe with data 
    :return: plot object
    """
    
    sc = ax.scatter(x=df['in_snr'], y=df['out_snr'], c=df['in_'+param], cmap='viridis', alpha=0.2)
    ax.plot(df['in_snr'], df['in_snr'])
    ax.set_ylabel('Recovered S/N')
    ax.grid()
    return sc


def snr_snr_plot(df_gt, df_op, gt_indices, op_indices, params, title = None, save=False):
    """
    Generates snr scatter plot for all the parameters 
    in the input data
    :param df_gt: dataframe with ground truth info
    :param df_op: dataframe with detected candidate info
    :param gt_indices: Ground truth indexes of candidates 
                       that were detected by the search soft
    :param op_indices: Output indexes of the candidates 
                       reported by the search soft
    :param params: list of parameters to plot
    :param title: Title of the plot
    :param save: To save the figure
    """
    
    df_scatter_plot = pd.concat([df_op.iloc[op_indices].reset_index(drop=True), 
                df_gt.iloc[gt_indices].reset_index(drop=True)], axis=1)    
    sc = []
    
    fig, ax = plt.subplots(len(params), 1, sharex=True)
    if len(params) == 1:
        ax = [ax]
    for i in range(len(params)):
        sc.append(snr_scatter(params[i], ax=ax[i], fig=fig, df=df_scatter_plot))
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(sc[i], cax=cax)
        if 'dm' in params[i]:
            label = 'DM'
        elif 'width' in params[i]:
            label = 'Width (s)'
        else:
            label = params[i]
        cbar.set_label(label, rotation=270)

    ax[len(params)-1].set_xlabel('Injected S/N')
    if title:
        plt.suptitle(f'{title}', y=0.93)
    fig.subplots_adjust(right=0.8, hspace=0.1)
    
    if save:
        figname = title+'_snr_snr_plot' if title else 'snr_snr_plot' 
        plt.savefig(f'{figname}.png', bbox_inches='tight')
        

def recall_1d(df_gt, gt_indices, param, recall_bins = 10, hist_bins = 30, title=None, save=False):
    """
    Generates the 1D recall plot with equal number 
    of examples in each bin, overlayed with the 
    ground truth histogram, for a given parameter
    :param df_gt: dataframe with ground truth info
    :param gt_indices: Ground truth indexes of candidates 
                       that were detected by the search soft
    :param param: parameter to plot
    :param recall_bins: number of bins for recall plot
    :param hist_bins: number of bins for param histogram
    :param title: Title of the plot
    :param save: To save the figure
    :returns: axis object of the plot 

    """
    
    df_out = df_gt.iloc[gt_indices]
    bins = histedges_equalN(df_gt[f'in_{param}'], recall_bins)
    gt_hist, _ = np.histogram(df_gt[f'in_{param}'], bins)
    out_hist, _ = np.histogram(df_out[f'in_{param}'], bins)
    
    recall = out_hist/gt_hist
    
    bin_mid = (bins[:-1] + bins[1:])/2

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(f'{param}')
    ax1.set_ylabel('Recall')
    ax1.step(bin_mid, recall, color='k')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number')
    ax2.hist(df_gt[f'in_{param}'], alpha=0.5, bins=hist_bins)
    ax2.tick_params(axis='y')

    if 'snr' in param or 'width' in param:
        ax1.set_xscale('log')
    
    ax1.grid()
    if title:
        plt.suptitle(f'{title}', y=1.01)
    fig.tight_layout()    
    if save:
        figname = title+f'_1d_recall_{param}' if title else f'1d_recall_{param}' 
        plt.savefig(f'{figname}.png', bbox_inches='tight')
    plt.show()

    return ax1


def manage_input(file):
    """
    Reads the json files generates by EYRA
    Benchmark scripts and generates ground
    truth and output dataframes with nice column
    names
    :param file: filepath of the json file to read
    :return df_gt: dataframe with ground truth info
    :return df_op: dataframe with detected candidate info
    :return gt_indices: Ground truth indexes of candidates 
                       that were detected by the search soft
    :return op_indices: Output indexes of the candidates 
                       reported by the search soft
    """
    with open(file, 'r') as f:
        data = json.load(f)
    
    dfgt = pd.DataFrame(data['ground_truth']['data'], columns=data['ground_truth']['column_names'])
    dfop = pd.DataFrame(data['implementation_output']['data'], columns=data['implementation_output']['column_names'])
    
    df_op = dfop[['DM (Dispersion measure)', 'SN (Signal to noise)', 'time (Time of arrival (s))']]
    df_op.columns = ['out_dm', 'out_snr', 'out_toa']
    
    df_gt = dfgt[['DM (Dispersion measure)', 'SN (Signal to noise)',
       'time (Time of arrival (s))','width_i (Width_i)', 'with_obs (Width_obs)',
       'spec_ind (Spec_ind)']]
    df_gt.columns = ['in_dm', 'in_snr', 'in_toa', 'in_width', 'in_width_obs', 'in_si']
    
    op_indices = []
    gt_indices = []
    for out_index in data['matches']:
        op_indices.append(int(out_index))
        gt_indices.append(data['matches'][out_index])
        
    return df_gt, df_op, gt_indices, op_indices

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Generates plots to visualise search software performance",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('-f', '--file', help='json file', type=str, required=True)
#     parser.add_argument('-ss', '--snr_snr_plot', help='Save snr snr plot', action='store_true')
#     parser.add_argument('-r', '--recall_plot', help='Save 1D recall plot', action='store_true')
#     inputs = parser.parse_args()
    
#     title = os.path.splitext(inputs.file)[0].split('_')[-1]
#     df_gt_plot, df_op_plot, gt_indices, op_indices = manage_input(inputs.file)
    
#     if inputs.snr_snr_plot:
#         snr_snr_plot(df_gt_plot, df_op_plot, gt_indices, op_indices, ['dm', 'width', 'toa'], title = None, save=True)    
    
#     if inputs.recall_plot:
#         recall_1d(df_gt_plot, gt_indices, 'dm', recall_bins = 10, hist_bins = 30, title=title, save=True)