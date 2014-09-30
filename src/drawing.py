import itertools
import networkx as nx
import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt

def draw_graph(title, G):
    """
    Draw the graph, the node size will be based on the number of lines squared 
    the character spoke in that scene, indicating their dominance/activity.
    
    The strength of the edges are based on the 
    """
    
    #print 'json repr:', node_link_data(G)
    #if logger.isEnabledFor(logging.DEBUG):
    #print 'nlines: %s', [(n, G.node[n]['nlines']) for n in G]

    plt.title(str(title))
    plt.xticks([])
    plt.yticks([])
    
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.spectral_layout(G)
    
    #node_size = [int(G.node[n]['nlines'])**2 for n in G]
    #node_size = [int(G.node[n]['nlines']) for n in G]

    c = 'b'
    nx.draw_networkx_edges(G, pos, edge_color=c, width=1, alpha=0.5)
    #nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c, alpha=0.5)
    # not sure what this is for...
    #nx.draw_networkx_nodes(G, pos, node_size=5, node_color='k')

    # would be good to have a little "judder" b/n nodes
    # need to fix the label spacing, or should figure it 
    # out based on the size of the entire graph...
    pos_lbls = pos.copy()
    if len(pos_lbls) > 1:
        for k in pos_lbls.keys():
            pos_lbls[k] += [0.01, 0.02]
            #pos_lbls[k] += [0.05, 0.03]

    nx.draw_networkx_labels(G, pos_lbls, alpha=0.5)
    plt.show()

def plotGMM(X_train, y_train, 
            #X_test, y_test, 
            samples, targets, lbls):
    from sklearn import mixture
    n_classes = len(np.unique(y_train))
    classifiers = dict((covar_type, 
                        mixture.GMM(n_components=n_classes,
                                    covariance_type=covar_type, 
                                    init_params='wc', 
                                    n_iter=20))
                       for covar_type in ['spherical', 'diag', 'tied', 'full'])
    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * n_classifiers / 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                        left=.01, right=.99)
    
    lbl_clrs = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                      for i in xrange(n_classes)])
    
        # Train the other parameters using the EM algorithm.
        classifier.fit(X_train)
    
        h = plt.subplot(2, n_classifiers / 2, index + 1)
        for n, (covars, color) in enumerate(zip(classifier._get_covars(), lbl_clrs)):
            make_ellipse(h, 
                         covars[:2, :2], 
                         classifier.means_[n, :2], 
                         color)

        for n, (lbl, color) in enumerate(zip(lbls, lbl_clrs)):
            data = samples[targets == n]
            # Just using the first 2 dims to plot X and Y
            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color, label=lbl)

        # Plot the test data with crosses
#         for n, color in enumerate(lbl_clrs):
#             data = X_test[y_test == n]
#             plt.plot(data[:, 0], data[:, 1], 'x', color=color)
    
#         for n, color in enumerate('rgb'):
#             data = iris.data[iris.target == n]
#             plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
#                         label=iris.target_names[n])
#         for n, color in enumerate('rgb'):
#             data = X_test[y_test == n]
#             plt.plot(data[:, 0], data[:, 1], 'x', color=color)
    
        y_train_pred = classifier.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes)
    
#         y_test_pred = classifier.predict(X_test)
#         test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#         plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
#                  transform=h.transAxes)
    
        plt.xticks(())
        plt.yticks(())
        plt.title(name)
    
    plt.legend(loc='lower right', prop=dict(size=12))
    plt.show()

def make_ellipse(ax, covars, means, color):
    # v - eigenvectors, w - eigenvalues
    v, w = np.linalg.eigh(covars)
    u = w[0] / np.linalg.norm(w[0])
    print 'After normalization...'
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v *= 9
    ell = mpl.patches.Ellipse(means, v[0], v[1],
                              180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)


def plotGMM2(gmm, X):
    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    #for i, (clf, title) in enumerate([(gmm, 'GMM'),
    #                              (dpgmm, 'Dirichlet Process GMM')]):
    #clf = gmm
    #splot = plt.subplot(2, 1, 1 + i)
    splot = plt.subplot(2, 1, 1)
    print 'Before predict...'
    Y_ = gmm.predict(X)
    print 'About to iterate...'
#     for i, (mean, covar, color) in enumerate(zip(
#             gmm.means_, gmm._get_covars(), color_iter)):
    for i, (mean, covar, color) in enumerate(zip(
            gmm.means_, gmm._get_covars(), color_iter)):
        print 'Iteration', i

        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        
        make_ellipse(splot, covar, mean, color)
#         v,w = linalg.eigh(covar)
#         u = w[0] / linalg.norm(w[0])
#         # Plot an ellipse to show the Gaussian component
#         angle = np.arctan(u[1] / u[0])
#         angle = 180 * angle / np.pi  # convert to degrees
#         ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
#         ell.set_clip_box(splot.bbox)
#         ell.set_alpha(0.5)
#         splot.add_artist(ell)
 
    #plt.xlim(-10, 10)
    #plt.ylim(-3, 6)
    plt.xticks(())
    plt.yticks(())
    #plt.title(title)
    plt.show()
