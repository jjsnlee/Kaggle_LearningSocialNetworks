import scipy as sp
from scipy.misc import comb
from itertools import product
from pymc import  Normal, Bernoulli, InvLogit, MCMC,MAP,deterministic

"""
http://socialabstractions.tumblr.com/post/53391947460/exponential-random-graph-models-in-python

For readers unfamiliar with ERGM, it is a modeling framework for network or graph data. 
An unfortunate fact of statistical inference on networks is that the independence assumptions 
of ordinary least squares are violated in deep, complex and interconnected ways (one of the 
core insights of social network analysis is that whether or not I am friends with you is 
tightly related to whether or not I am friends with your friends). ERGMs attempt to bring a 
linear modeling approach to network estimation by assuming that these inherent interdependencies 
depend mostly on network characteristics that a savvy modeler can explicitly specify. If you 
have reason to believe that the major network dependencies in your data can be controlled for 
by reciprocity and transitivity, for instance, you can simply include those terms in your model 
specification and hope (read: assume) that the rest of your errors are independent. While they 
are not perfect, ERGMs have become a widely accepted method among social network analysts.

Most often, ERGMs are estimated using the ergm package for the R statistical environment. 
This is a wonderful package put together by many of the top researchers in the field of 
network inference. Recently I needed to estimate a model that allowed more low-level control 
of the modelling than this package allowed however, so I turned to PyMC to see if I could 
implement ERGM estimation myself. PyMC is an invaluable python package that makes Markov-chain 
Monte-Carlo (MCMC) estimation straightforward and, importantly, very fast to implement. Getting 
estimates for pretty much any model in PyMC takes barely any more work than just specifying 
that model, and it's well designed enough that writing your own step methods (and even sampling 
algorithms) in python is a snap.
"""

# functions to get delta matrices
def mutualDelta(am):
    return(am.copy().transpose())

def istarDelta(am,k):
    if k == 1:
        # if k == 1 then this is just density
        res = sp.ones(am.shape)
        return(res)
    res = sp.zeros(am.shape,dtype=int)
    n = am.shape[0]
    for i,j in product(xrange(n),xrange(n)):
        if i!=j:
            nin = am[:,j].sum()-am[i,j]
            res[i,j] = comb(nin,k-1,exact=True)
    return(res)

def ostarDelta(am,k):
    if k == 1:
        # if k == 1 then this is just density
        res = sp.ones(am.shape)
        return(res)
    res = sp.zeros(am.shape,dtype=int)
    n = am.shape[0]
    for i,j in product(xrange(n),xrange(n)):
        if i!=j:
            nin = am[i,:].sum()-am[i,j]
            res[i,j] = comb(nin,k-1,exact=True)
    return(res)


def makeModel(adjMat):

    # define and name the deltas
    termDeltas = {
        'deltamutual':mutualDelta(adjMat),
        'deltaistar1':istarDelta(adjMat,1),
        'deltaistar2':istarDelta(adjMat,2),
        'deltaistar3':istarDelta(adjMat,3),
        'deltaostar2':ostarDelta(adjMat,2)
    }

    # create term list with coefficients
    termList = []
    coefs = {}
    for dName,d in termDeltas.items():
        tName = 'theta'+dName[5:]
        coefs[tName] = Normal(tName,0,0.001,value=sp.rand()-0.5)
        termList.append(d*coefs[tName])

    # get individual edge probabilities
    @deterministic(trace=False,plot=False)
    def probs(termList=termList):
        probs = 1./(1+sp.exp(-1*sum(termList)))
        probs[sp.diag_indices_from(probs)]= 0
        return(probs)

    # define the outcome as 
    outcome = Bernoulli('outcome',probs,value=adjMat,observed=True)

    return(locals())


if __name__ == '__main__':
    # load the prison data
    with open('prison.dat','r') as f:
        rowList = list()
        for l in f:
            rowList.append([int(x) for x in l.strip().split(' ')])
        adjMat = sp.array(rowList)
    
    # make the model as an MCMC object
    m = makeModel(adjMat)
    mc = MCMC(m)

    # estimate
    mc.sample(30000,1000,50)
