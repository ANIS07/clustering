import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
import igraph as ig
import difflib
from operator import itemgetter


def generate_fake_data(number_code,number_treatment,number_provider,treatment_column,provider_column,code_column):
    
    # Define traitement , specialy and providers code
    speciality_code = []
    for i in range(number_code):
        speciality_code.append('C'+str(i))

    treatment_code = []
    for i in range(number_treatment):
        treatment_code.append('T'+str(i))

    provider_code = []
    for i in range(number_provider):
        provider_code.append('P'+str(i))
    
    # construct fake data : unbalanced
    data_base = []
    count = 0
    for treatment in treatment_code:
        count+=1
        if count<number_treatment/4 :
            spec = speciality_code[random.randint(0, number_code/4)]
            prov = provider_code[random.randint(0, number_provider/4)]

        elif number_treatment/4<=count and count<number_treatment/2:
            spec = speciality_code[random.randint(number_code/4, number_code/2)]
            prov = provider_code[random.randint(number_provider/4, number_provider/2)]

        elif count<number_treatment/2<=count and count<3*number_treatment/4:
            spec = speciality_code[random.randint(number_code/2, 3*number_code/4)]
            prov = provider_code[random.randint(number_provider/2, 3*number_code/4)]

        else:
            spec = speciality_code[random.randint(3*number_code/4,number_code-1)]
            prov = provider_code[random.randint(3*number_provider/4,number_provider-1)]
        data_base.append({treatment_column:treatment,provider_column:prov,code_column:spec})
    return pd.DataFrame(data_base)


def bag_of_words_similarity(traitement):
    """
    Computes the similarity matrix from traitement (seen as list of traitement) 
    via bag-of-words approach

    :param traitement: list of lists of traitement
    :type traitement: list
    :return: similarity_matrix
    :rtype: an array of size nb_trips * nb_trips
    """

    # Bag of words representation + Tf-Idf
    tfidf_matrix = TfidfVectorizer(analyzer='word', preprocessor=None, smooth_idf=True).fit_transform(traitement)

    # Get the similarities of all the first parts of trips, then the second.
    cosine_sim= cosine_similarity(tfidf_matrix)


    # Put the diagonal to zero
    n = len(cosine_sim)
    cosine_sim[range(n), range(n)] = 0

    return cosine_sim

def ratcliff_similarity(traitement):
    """
    Computes the similarity matrix from traitement (seen as list of traitement)
    using Ratcliff similarity

    :param traitement: list of lists of traitement
    :type traitement: list
    :return: similarity_matrix
    :rtype: an array of size nb_trips * nb_trips
    """

    nb_provider = len(traitement)

    similarity_matrix = np.zeros((nb_provider, nb_provider))

    # Fill the similarity matrix
    for i in range(nb_provider):
        for j in range(nb_provider):
            if i < j:
                similarity = round(difflib.SequenceMatcher(None, traitement[i], traitement[j]).ratio(), 2)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    return similarity_matrix


def get_clusters(similarity_matrix,traitement,group_label,threshold):


    # Delete low similarities
    low_values_indices = similarity_matrix < threshold
    similarity_matrix[low_values_indices] = 0

    h = ig.Graph.Weighted_Adjacency((similarity_matrix > 0).tolist(), mode='undirected', attr='weight', loops=False)
    h.es['weight'] = similarity_matrix[similarity_matrix.nonzero()]
    h.vs['label'] = range(len(similarity_matrix))

    # Clustering (get the dendogram and select the clustering that maximize the modularity)
    dendogram = h.community_fastgreedy(weights=h.es["weight"]) 
    clustering = dendogram.as_clustering()
    dic_of_clusters = {}

    # Make a dictionary that summarizes the clustering (with trip ids, select the centers,...)
    for v,k in enumerate(sorted(range(len(clustering)), key=lambda k: len(clustering[k]), reverse=True)):
        if len(clustering[k]) > 1:

            # Select the center (Among the one-third with highest degree, select the longest)
            j = clustering.subgraph(k)
            labels = j.vs['label']
            degrees = [j.strength(i.index, loops=False, weights=j.es["weight"]) for i in j.vs]
            try:
                sizes = [len(x.split()) for x in traitement]
            except:
                sizes = [len(x) for x in traitement]

            top_degrees = sorted(zip(degrees, sizes, labels), reverse=True)[:len(labels)/3 + 2]
            center_label = max(top_degrees, key=lambda x: x[1])[2]

            # Fill the dictionary
            dic_of_clusters[v] = {'cluster': [group_label[l] for l in labels], 'center': group_label[center_label]}
    return dic_of_clusters