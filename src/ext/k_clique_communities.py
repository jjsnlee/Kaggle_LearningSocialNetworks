import networkx as nx
import pandas as pd

"""
From: https://www.kaggle.com/c/learning-social-circles/forums/t/10266/starter-code-to-beat-the-connected-components-benchmark
I'm attaching a starter code to beat the connected components benchmark.

It should take a few seconds to run, and achieve a score of 4822 on the public leaderboard. 
it has a few params that can be tuned so you might be able to squeeze some more out of this 
if you play with it a little.
"""

#%% some function definitions

def read_nodeadjlist(filename):
    G = nx.Graph()
    for line in open(filename):
        e1, es = line.split(':')
        es = es.split()
        for e in es:
            if e == e1: continue
            G.add_edge(int(e1),int(e))
    return G

#%% set script params

# folder paths
egonetFolderName = '../Data/egonets/'
submissionFolderName = '../Subs/'

# params
# the param settings that will beat the benchmark are:
def main(cliqueSize=5,
         tooLittleFriendsInCircleThreshold=10,
         tooManyNodesThreshold=220,
         submissionNumber=13):
    
    #%% make a submission
    submission = pd.read_csv(submissionFolderName + 'sample_submission.csv')
    for userId in list(submission['UserId']):

        # read graph
        filename = str(userId) + '.egonet'
        G = read_nodeadjlist(egonetFolderName + filename)
    
        # do not calculate for large graphs (it takes too long)
        if len(G.nodes()) > tooManyNodesThreshold:
            print 'skipping user ' + str(userId)
            continue
        else:
            print 'predicting for user ' + str(userId)

        listOfCircles = \
        predict_user(G, cliqueSize=cliqueSize,
                     tooLittleFriendsInCircleThreshold=tooLittleFriendsInCircleThreshold,
                     #tooManyNodesThreshold=tooManyNodesThreshold
                     )
        # populate prediction string
        predictionString = ''
        for circle in listOfCircles:
            for node in circle:
                predictionString = predictionString + str(node) + ' '
            predictionString = predictionString[:-1] + ';'
        predictionString = predictionString[:-1]

        # if no prediction was created, use 'all friends in one circle'
        if len(listOfCircles) > 0:
            submission.ix[submission['UserId'] == userId,'Predicted'] = predictionString

    submission.to_csv(submissionFolderName + str(submissionNumber) + '.csv', index=False)

def predict_user(G,
                 cliqueSize=5,
                 tooLittleFriendsInCircleThreshold=10,
                 #tooManyNodesThreshold=220
                 ):
    # find comunities using k_clique_communities()
    listOfCircles = []
    kCliqueComunities = list(nx.k_clique_communities(G,cliqueSize))
    for community in kCliqueComunities:
        # leave only relativly large communities
        if len(community) >= tooLittleFriendsInCircleThreshold:
            listOfCircles.append(list(community))

    return listOfCircles
