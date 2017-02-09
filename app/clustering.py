import manager
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.utils import shuffle
import community.community_louvain as com
import networkx as nx
import numpy as np
import operator
import sys, os
import matplotlib.pyplot as plt
from math import log10
import copy
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import pickle

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

    
class Cluster:
    """
    Compute the clusters with the knn-graph based clustering using Louvain aglorithm.
    
    Parameters
    ----------
    name : string, optional
        a name for the cluster (use it to store the experiment configurations)
    basket : manager.Basket
        a basket holding the sound collection to cluster
    k_nn : int
        the parameter of the k nearest neighbour for graph generation. Default to 20
      
    Examples
    --------
    from knn_graph_clustering import *
    c = manager.Client()
    b = c.load_basket_pickle('UrbanSound8K')
    cluster = Cluster(basket=b)
    cluster.run()
    
    """
    def __init__(self, name='Cluster Object', basket=None, k_nn=20):
        self.name = name
        self.basket = basket
        self.k_nn = k_nn
        self.feature_type = None
        self.acoustic_features = None
        self.acoustic_similarity_matrix = None
        self.text_features = None
        self.text_similarity_matrix = None
        self.graph = None
        self.graph_knn = None
        self.nb_clusters = None
        self.ids_in_clusters = None
    
    def run(self, k_nn=None):
        """Run all the steps for generating cluster (by default with text features)"""
        if k_nn:
            self.k_nn = k_nn
        if not(isinstance(self.text_similarity_matrix, np.ndarray)) and not(isinstance(self.acoustic_similarity_matrix, np.ndarray)): # do not calculate again the similarity matrix if it is already done
            self.compute_similarity_matrix()
        if not(self.graph_knn == self.k_nn): # do not generate graph it is already done with the same k_nn parameter
            self.generate_graph()
        self.cluster_graph()
        self.create_cluster_baskets()
        self.display_clusters()
        if hasattr(self.basket, 'clas'): # some baskets have a clas attribute where are stored labels for each sound instance
            self.evaluate()
    
    # __________________ FEATURE __________________ #
    def compute_similarity_matrix(self, basket=None, feature_type='text'):
        """
        feature_type : 'text' or 'acoustic'
        the type of features used for computing similarity between sounds. 
        """
        self.feature_type  = feature_type
        basket = basket or self.basket
        if basket == None:
            print 'You must provide a basket as argument'
        else:
            if feature_type == 'text':
                self.extract_text_features(basket)
                self.create_similarity_matrix_text(self.text_features)
            elif feature_type == 'acoustic':
                self.extract_acoustic_features(basket)
                self.create_similarity_matrix_acoustic(self.acoustic_features)
            print '\n\n >>> Similarity Matrix Computed <<< '
                
    def extract_text_features(self, basket=None):
        basket = basket or self.basket
        preproc_basket = copy.deepcopy(basket)
        t = preproc_basket.preprocessing_tag() #some stemming 
        for idx, tt in enumerate(t):
            preproc_basket.sounds[idx].tags = tt
        nlp = manager.Nlp(preproc_basket) # counting terms...
        nlp.create_sound_tag_matrix() # create the feature vectors
        self.text_features = nlp.sound_tag_matrix
        
    def create_similarity_matrix_text(self, features=None):
        if features == None:
            features = self.text_features
        if features == None:
            print 'You must provide the text features as argument or run extract_text_features() first'
        else:
            self.text_similarity_matrix = cosine_similarity(features)
        
    def extract_acoustic_features(self, basket=None):
        """Extract acoustic features"""
        basket = basket or self.basket
        basket.analysis_stats = [None] * len(self.basket) # is case of the basket is old, now analysis_stats contains None values initialy
        basket.add_analysis_stats()
        basket.remove_sounds_with_no_analysis()
        self.acoustic_features = basket.extract_descriptor_stats(scale=True) # list of all descriptors stats for each sound in the basket
    
    def create_similarity_matrix_acoustic(self, features=None):
        if features == None:
            features = self.text_features
        if features == None:
            print 'You must provide the acoustic features as argument or run extract_acoustic_features() first'
        else:
            matrix = euclidean_distances(features)
            matrix = matrix/matrix.max()
            self.acoustic_similarity_matrix = 1 - matrix
            
    # __________________ GRAPH __________________ #
    def generate_graph(self, similarity_matrix=None, k_nn=None):
        blockPrint()
        k_nn = k_nn or self.k_nn
        if similarity_matrix == None:
            if self.feature_type == 'text':
                similarity_matrix = self.text_similarity_matrix
            elif self.feature_type == 'acoustic':
                similarity_matrix = self.acoustic_similarity_matrix
        print similarity_matrix
        self.graph = self.create_knn_graph(similarity_matrix, k_nn)
        enablePrint()
        self.graph_knn = k_nn #save the k_nn parameters
        print '\n >>> Graph Generated <<< '
        
    def cluster_graph(self, graph=None):
        graph = graph or self.graph
        classes = com.best_partition(graph)
        self.nb_clusters = max(classes.values()) + 1
        self.dendrogram = com.generate_dendrogram(graph)
        self.ids_in_clusters = [[e for e in classes.keys() if classes[e]==cl] for cl in range(self.nb_clusters)]
        print '\n >>> Graph Clustered <<<\n Found %d clusters'%self.nb_clusters
        
    @staticmethod
    def nearest_neighbors(similarity_matrix, idx, k):
        distances = []
        for x in range(len(similarity_matrix)):
            distances.append((x,similarity_matrix[idx][x]))
        distances.sort(key=operator.itemgetter(1), reverse=True)
        return [d[0] for d in distances[0:k]]
    
    def create_knn_graph(self, similarity_matrix, k):
        """ Returns a knn graph from a similarity matrix - NetworkX module """
        np.fill_diagonal(similarity_matrix, 0) # for removing the 1 from diagonal
        g = nx.Graph()
        g.add_nodes_from(range(len(similarity_matrix)))
        for idx in range(len(similarity_matrix)):
            g.add_edges_from([(idx, i) for i in self.nearest_neighbors(similarity_matrix, idx, k)])
            print idx, self.nearest_neighbors(similarity_matrix, idx, k)
        return g
    
    # __________________ DISPLAY __________________ #
    def create_cluster_baskets(self):
        list_baskets = [self.basket.parent_client.new_basket() for i in range(self.nb_clusters)]
        for cl in range(len(self.ids_in_clusters)):
            for s in self.ids_in_clusters[cl]:
                list_baskets[cl].push(self.basket.sounds[s])
        self.cluster_baskets = list_baskets
        print '\n >>> Basket for each clusters created <<< '
        
    def display_clusters(self):
        tags_occurrences = [basket.tags_occurrences() for basket in self.cluster_baskets]
        normalized_tags_occurrences = []
        for idx, tag_occurrence in enumerate(tags_occurrences):
            normalized_tags_occurrences.append([(t_o[0], float(t_o[1])/len(self.cluster_baskets[idx].sounds)) for t_o in tag_occurrence])
        self.tags_oc = normalized_tags_occurrences
        
        def print_basket(list_baskets, normalized_tags_occurrences, num_basket, max_tag = 20):
            """Print tag occurrences"""
            print '\n Cluster %s, containing %s sounds' % (num_basket, len(list_baskets[num_basket])) 
            for idx, tag in enumerate(normalized_tags_occurrences[num_basket]):
                if idx < max_tag:
                    print tag[0].ljust(30) + str(tag[1])[0:5]
                else:
                    break
        
        print '\n\n'
        print '\n ___________________________________________________________'
        print '|_________________________RESULTS___________________________|'
        print '\n Cluster tags occurrences for Tag based method (normalized):'
            
        for i in range(len(self.ids_in_clusters)):
                print_basket(self.cluster_baskets, normalized_tags_occurrences, i, 10)
        
    def plot(self):
        nx.draw(self.graph)
        plt.show()

    def evaluate(self):
        # the basket needs the hidden clusters information
        # basket.clas = [clas_sound_1, clas_sound_2, ...]
        all_clusters, all_hidden_clusters = construct(self, self.basket)
        self.score = homogeneity(all_clusters, all_hidden_clusters)
        print '\n\n' 
        print 'Homogeneity = %s, k_nn = %s' %(self.score,self.k_nn)
        
        
# __________________ EVALUATION __________________ #
def construct(cluster, b):
    all_clusters = cluster.ids_in_clusters
    all_hidden_clusters = []
    for cl in range(max(flat_list(b.clas))+1): 
        clust = []
        for idx, c in enumerate(b.clas):
            if int(c) == cl:
                clust.append(idx)
        all_hidden_clusters.append(clust)
    return all_clusters, all_hidden_clusters

def my_log(value):
    if value == 0:
        return 0
    else:
        return log10(value)

def purity(cluster, all_hidden_clusters):
    """ Calculate the purity of a cluster """
    purity = 0.
    for hidden_cluster in all_hidden_clusters:
        proba = prob(cluster, hidden_cluster)
        purity -= proba*my_log(proba)
    return purity

def prob(cluster, hidden_cluster):
    """ Calculate the probability of hidden_cluster knowing cluster """
    return len(intersec(cluster, hidden_cluster))/float(len(cluster))

def intersec(list1, list2):
    """ Intersection of two lists """
    return list(set(list1).intersection(set(list2)))

def flat_list(l):
    """ Convert a nested list to a flat list """
    return [item for sublist in l for item in sublist]
    
def homogeneity(all_clusters, all_hidden_clusters):
    """ Caculate the homogeneity of the found clusters with respect to the hidden clusters. Based on Entropy measure """
    total = 0.
    for cluster in all_clusters:
        total += len(cluster) * purity(cluster, all_hidden_clusters)
    total = total / (log10(len(all_hidden_clusters)) * len(flat_list(all_clusters)))
    total = 1. - total
    return total
    
    
# __________________ W2V __________________ #
class W2v:
    # feature vectors amd similarity matrix with w2v
    def __init__(self, name='Cluster Object', basket=None, size_w2v=20):
        self.name = name
        self.basket = basket
        self.size_w2v = size_w2v
        
    def run(self):
        X = np.array(self.basket.preprocessing_tag_description())
        #X = shuffle(X, random_state=0)
        #w2v_model = pickle.load(open('/home/xavier/Documents/dev/freesound-python/w2v_freesoundDb.pkl','rb'))
        w2v_model = Word2Vec(X, size=self.size_w2v, window = 1000, min_count = 5, workers = 4, sg = 0)
        term_vectors = {w: vec for w, vec in zip(w2v_model.index2word, w2v_model.syn0)}
        tfidfEmbeddor = self.TfidfEmbeddingVectorizer(term_vectors).fit(X)
        self.sound_vectors = tfidfEmbeddor.transform(X)
        matrix = euclidean_distances(self.sound_vectors)
        self.similarity_matrix = 1 - matrix/matrix.max()
        cluster = Cluster(basket=self.basket)
        cluster.text_similarity_matrix = self.similarity_matrix
        cluster.feature_type = 'text'
        return cluster
    
    class MeanEmbeddingVectorizer(object):
        def __init__(self, word2vec):
            self.word2vec = word2vec
            self.dim = len(word2vec.itervalues().next())

        def fit(self, X, y):
            return self 

        def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])

    # and a tf-idf version of the same
    class TfidfEmbeddingVectorizer(object):
        def __init__(self, word2vec):
            self.word2vec = word2vec
            self.word2weight = None
            self.dim = len(word2vec.itervalues().next())

        def fit(self, X):
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(X)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of 
            # known idf's
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(
                lambda: max_idf, 
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

            return self

        def transform(self, X):
            return np.array([
                    np.mean([self.word2vec[w] * self.word2weight[w]
                             for w in words if w in self.word2vec] or
                            [np.zeros(self.dim)], axis=0)
                    for words in X
                ]) 
