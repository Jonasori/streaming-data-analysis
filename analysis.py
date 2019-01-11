"""
Some exploratory data analysis on a provided dataset.
"""

import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sp
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


sns.set_style("white")
cmap = sns.diverging_palette(220, 20, center='dark', as_cmap=True)
palette = sns.diverging_palette(220, 20, center='dark', as_cmap=False)
cmap_light = sns.diverging_palette(220, 20, center='light', as_cmap=True)

data_path = '/Users/jonas/Desktop/Programming/Python/streaming_data_analysis/'
figure_path = data_path + 'figures/'


def get_data():
    df = pd.read_csv(data_path + 'data.csv').dropna()
    df['isp'].replace('Datch Telecam', 'DT', inplace=True)

    # Add in some extra relationships
    total_data = pd.Series(df['p2p'] + df['cdn'])
    df['total_data'] = total_data
    p2p_frac = pd.Series(df['p2p']/df['total_data'])
    cdn_frac = pd.Series(df['cdn']/df['total_data'])
    df['p2p_frac'] = p2p_frac
    df['cdn_frac'] = cdn_frac
    df['relative_use'] = df['p2p'] / df['cdn']
    return df
df = get_data()



# Make some subsets that will be useful for playing with.
streams = [df[ df['stream_id'] == id ].reset_index()
           for id in list(df.stream_id.unique())]

browsers = [df[ df['browser'] == id ].reset_index()
            for id in list(df.browser.unique())]

isps = [df[ df['isp'] == id ].reset_index()
        for id in list(df.isp.unique())]

df_not_connected = df[ df['connected'] == False ]
df_connected = df[ df['connected'] == True ]






### PLOTTERS ###


def plot_heatmaps(df, save=False):
    """Plot six heatmaps across three categorical vars and two data units."""
    plt.close()

    # Super janky but makes things simple for iteration.
    groups = ['browser', 'isp', 'stream_id', 'browser']

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
    for i in range(3):  # 3 groups
        x_axis, y_axis = (groups[i], groups[i + 1])
        df_wide=df.pivot_table( index=x_axis, columns=y_axis, values='total_data' )
        sns.heatmap(df_wide, ax=ax[0][i], cmap='Blues')

        df_wide=df.pivot_table( index=x_axis, columns=y_axis, values='p2p_frac' )
        sns.heatmap(df_wide, ax=ax[1][i], cmap='Reds')


    ax[0][1].set_title('Total Data Used', weight='bold')
    ax[1][1].set_title('P2P Data Transfered/Total Transfered', weight='bold')

    # sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(figure_path + 'usage-heatmap.png', dpi=200)
    plt.show()
plot_heatmaps(df, save=True)



def plot_bars(df, save=False):
    """Make some bar plots. Not useful."""
    plt.close()

    groups = ['browser', 'isp', 'stream_id', 'browser']
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
    for i in range(3):
        sns.barplot(x=groups[i], y='total_data', data=df, ax=ax[0][i],
                    palette=sns.diverging_palette(220, 20, center='dark'))

        sns.barplot(x=groups[i], y='p2p_frac', data=df, ax=ax[1][i],
                    palette=sns.diverging_palette(220, 20, center='dark'))

    ax[0][1].set_title('Total Data Used', weight='bold')
    ax[1][1].set_title('P2P Data Transfered/Total Transfered', weight='bold')
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(figure_path + 'usage-bars.png', dpi=200)
    plt.show()
plot_bars(df, save=True)



def plot_stream_distributions(df, grouping='stream_id', color_grouping=None):
    """Make a gif of how different members of a group draw their data."""
    plt.close()

    x_axis, y_axis = 'cdn', 'p2p'
    try:
        colors = sns.diverging_palette(220, 20, center='dark',
                                       n=len(list(df[color_grouping].unique())))
    except ValueError:
        color='darkred'
    images = []
    for i in range(len(list(df[grouping].unique()))):
        id = list(df[grouping].unique())[i]
        df_stream_sub = df[ df[grouping] == id ].reset_index()

        if not color_grouping:
            plt.scatter(df_stream_sub[x_axis], df_stream_sub[y_axis],
                        marker='.', color=color, alpha=0.1)

        else:
            sns.lmplot(x=x_axis, y=y_axis, data=df_stream_sub,
                       hue=color_grouping, markers='.', scatter_kws={'alpha':0.3},
                       fit_reg=False, legend=False, palette=colors)
            plt.legend(loc='lower left')


        xticks = np.linspace(min(df_stream_sub[x_axis]),
                             max(df_stream_sub[x_axis]), 4)
        yticks = np.linspace(min(df_stream_sub[y_axis]),
                             max(df_stream_sub[y_axis]), 4)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel(x_axis.upper(), weight='bold')
        plt.ylabel(y_axis.upper(), weight='bold')

        title = id if grouping.lower() != 'stream_id' else 'Stream ' + str(id)
        plt.title(title, weight='bold')

        filename = figure_path + 'stream{}-kde.png'.format(str(id))
        plt.savefig(filename)
        plt.close()

        print "Finished stream " + str(id)
        images.append(imageio.imread(filename))

    # Make them into a little gif
    postfix = '-color-grouped' if color_grouping else ''
    figure_name = figure_path + 'data-distributions-by_{}.gif'.format(grouping + postfix)
    imageio.mimsave(figure_name, images, duration=1)
    print "Finished making gif"

    for id in list(df[grouping].unique()):
        filename = figure_path + 'stream{}-kde.png'.format(str(id))
        sp.call(['rm', '-rf', '{}'.format(filename)])

    print "All done!"
plot_stream_distributions(df_connected, grouping='isp', color_grouping=None)




def plot_traffic_by_metric(df, data_metric='total', save=False):
    """Plot a triptych of two bar plots and a heatmap. Not useful."""
    if data_metric.lower() == 'total':
        y_axis = 'total_data'
        title_prefix = 'Total'
    elif data_metric.lower() == 'relative':
        y_axis = 'relative_use'
        title_prefix = 'Relative'

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

    sns.barplot(x='browser', y=y_axis, data=df, ax=ax[0],
                palette=sns.diverging_palette(220, 20, center='dark'))

    sns.barplot(x='isp', y=y_axis, data=df, ax=ax[1],
                palette=sns.diverging_palette(220, 20, center='dark'))

    df_wide=df.pivot_table( index='isp', columns='browser', values=y_axis )
    sns.heatmap(df_wide, ax=ax[2], cmap=cmap)


    ax[0].set_title(title_prefix + ' Data Use by Browser', weight='bold')
    ax[1].set_title(title_prefix + ' Data Use by ISP', weight='bold')
    ax[2].set_title(title_prefix + ' Data Use by Both', weight='bold')
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(figure_path + title_prefix + '-Usage.png', dpi=200)
    plt.show()
# plot_traffic_by_metric(df, data_metric='total', save=False)






### PLAYGROUND ###
"""
sns.lmplot(x='cdn', y='p2p', data=streams[3],
           hue='isp', markers='.', scatter_kws={'alpha':1},
           fit_reg=False, legend=True, palette=palette)

plt.show()
"""



# Simple ML Stuff:

def k_means(df, truncate=True, n=2):

    test_df = df.drop(df.index[5000:]) if truncate else df
    X = np.array([np.array(test_df['cdn']), np.array(test_df['p2p'])]).T

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap=cmap)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

def spectral_clustering(df, truncate=True, n=3):

    test_df = df.drop(df.index[5000:]) if truncate else df
    X = np.array([np.array(test_df['cdn']), np.array(test_df['p2p'])]).T

    model = SpectralClustering(n_clusters=n, affinity='nearest_neighbors',
                               assign_labels='kmeans')
    labels = model.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels,
                s=50, cmap=cmap)

    plt.show()




# The End
