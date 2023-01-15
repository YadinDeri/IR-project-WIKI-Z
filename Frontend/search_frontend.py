from flask import Flask, request, jsonify
import re
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from inverted_index_body_gcp import *
from inverted_index_title_gcp import *
from inverted_index_anchor_gcp import *
import math
import io

bucket_name = 'ir-project-z'
client = storage.Client()
bucket = client.bucket(bucket_name)

# ----------------------- Import pkls --------------------------------------------

# --------------- Import Body pkl ---------------
index_src_body = "index_body.pkl"
blob_index_body = bucket.blob(f"{index_src_body}")
pickel_in_body = blob_index_body.download_as_string()
body_index = pickle.loads(pickel_in_body)

# --------------- Import Title pkl ---------------
index_src_title = "postings_gcp_title/index_title.pkl"
blob_index_title = bucket.blob(f"{index_src_title}")
pickel_in_title = blob_index_title.download_as_string()
title_index = pickle.loads(pickel_in_title)

# --------------- Import Anchor pkl ---------------
index_src_anchor = "postings_gcp_anchor/index_anchor.pkl"
blob_index_anchor = bucket.blob(f"{index_src_anchor}")
pickel_in_anchor = blob_index_anchor.download_as_string()
anchor_index = pickle.loads(pickel_in_anchor)

# ----------------------------------------------------------------------------------
# ---------------------------Page Rank ---------------------------------------------
blob_page_rank = bucket.blob('page_rank/pagerank.csv')
content = blob_page_rank.download_as_string()
pr = pd.read_csv(io.StringIO(content.decode()), index_col=0, header=None, squeeze=True)
# # ----------------------------------------------------------------------------------
# # ---------------------------Page View ---------------------------------------------
file_name = 'page_view/pageviews.pkl'
blob = bucket.get_blob(file_name)
pkl_bytes = blob.download_as_string()
dict_page_view = pickle.loads(pkl_bytes)
# ----------------------------------------------------------------------------------

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["make", "how", "to", "category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
from nltk.metrics.association import ContingencyMeasures

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def read_posting_list(inverted, w, path, kind):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, kind)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    results = []
    query_str = request.args.get('query', '')
    if len(query_str) == 0:
        return jsonify(results)
    # BEGIN SOLUTION
    # Extract query words and remove stopwords
    query = [token.group() for token in re.finditer(r'\w+', query_str.lower()) if token.group() not in all_stopwords]
    path_to_bin = 'postings_gcp_title/'
    # Collect documents that contain query words
    query_binary_similarity = {}
    for word in query:
        try:
            posting_lst = read_posting_list(title_index, word, path_to_bin, 'title')
        except:
            continue
        if posting_lst:
            for doc in posting_lst:
                doc_id = doc[0]
                if doc_id in query_binary_similarity:
                    query_binary_similarity[doc_id] += 1
                else:
                    query_binary_similarity[doc_id] = 1
    # Return empty result if no documents contain query words
    if not query_binary_similarity:
        return []

    # Sort documents by query word frequency
    sorted_query_similarity = {k: v for k, v in
                               sorted(query_binary_similarity.items(), key=lambda x: x[1], reverse=True)}

    # Collect document names for matching documents
    for doc_id in sorted_query_similarity:
        if doc_id in title_index.doc_name:
            results.append((doc_id, title_index.doc_name[doc_id]))

    # END SOLUTION
    return jsonify(results[:80])

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    results = []
    query_str = request.args.get('query', '')
    if len(query_str) == 0:
        return jsonify(results)
    # BEGIN SOLUTION
    # tokenize the query and remove stopwords

    tokenized_query = [token.group() for token in RE_WORD.finditer(query_str.lower()) if
                       token.group() not in all_stopwords]

    word_weights_in_query = {}
    cosine_similarity = {}
    # calculate the weights of the words in the query
    word_weights = Counter(tokenized_query)
    for word in word_weights:
        word_weights_in_query[word] = word_weights[word] / len(tokenized_query)
    path_to_bin = 'postings_gcp_body/'
    # calculate the upper part of the cosine similarity formula
    product_for_bottom_from_query = 0
    for word in word_weights:
        weight = word_weights_in_query[word]
        product_for_bottom_from_query += math.pow(weight, 2)
        if word not in body_index.idf:
            continue
        idf = body_index.idf[word]
        word_posting_lst = read_posting_list(body_index, word, path_to_bin, "body")
        if len(word_posting_lst) > 0:
            for tup in word_posting_lst:
                if tup[0] not in cosine_similarity:
                    cosine_similarity[tup[0]] = (tup[1] / body_index.doc_len[tup[0]]) * idf * weight
                else:
                    cosine_similarity[tup[0]] += (tup[1] / body_index.doc_len[tup[0]]) * idf * weight

    # calculate the bottom part of the cosine similarity formula
    for doc_id in cosine_similarity:
        dominator = math.sqrt(body_index.dominator_mapping[doc_id] * product_for_bottom_from_query)
        cosine_similarity[doc_id] = cosine_similarity[doc_id] / dominator

    # sort the cosine similarity values and get the top 100 results
    sorted_cosine_similarity = {k: v for k, v in
                                sorted(cosine_similarity.items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for doc in sorted_cosine_similarity:
        if i >= 100:
            break
        results.append((doc, body_index.doc_name[doc]))
        i += 1

    # END SOLUTION
    return jsonify(results[:80])


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    results = []
    query_str = request.args.get('query', '')
    if len(query_str) == 0:
        return jsonify(results)
    # BEGIN SOLUTION
    # Extract query words and remove stopwords
    query = [token.group() for token in re.finditer(r'\w+', query_str.lower()) if token.group() not in all_stopwords]
    path_to_bin = 'postings_gcp_title/'
    # Collect documents that contain query words
    query_binary_similarity = {}
    for word in query:
        try:
            posting_lst = read_posting_list(title_index, word, path_to_bin, 'title')
        except:
            continue
        if posting_lst:
            for doc in posting_lst:
                doc_id = doc[0]
                if doc_id in query_binary_similarity:
                    query_binary_similarity[doc_id] += 1
                else:
                    query_binary_similarity[doc_id] = 1

    # Return empty result if no documents contain query words
    if not query_binary_similarity:
        return []

    # Sort documents by query word frequency
    sorted_query_similarity = {k: v for k, v in
                               sorted(query_binary_similarity.items(), key=lambda x: x[1], reverse=True)}

    # Collect document names for matching documents
    results = []
    for doc_id in sorted_query_similarity:
        if doc_id in title_index.doc_name:
            results.append((doc_id, title_index.doc_name[doc_id]))

    # END SOLUTION
    return jsonify(results[:80])


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = []
    temp = defaultdict(int)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # Calculate frequency of terms in query
    path_to_bin = 'postings_gcp_anchor/'
    binary_similarity = {}
    for word in list_of_tokens:
        try:
            posting_list = read_posting_list(anchor_index, word, path_to_bin, 'anchor')
        except:
            continue

        if len(posting_list) > 0:
            for doc in posting_list:
                if doc[0] in binary_similarity:
                    binary_similarity[doc[0]] += 1
                else:
                    binary_similarity[doc[0]] = 1

    sorted_si = {k: v for k, v in sorted(binary_similarity.items(), key=lambda i: i[1], reverse=True)}
    for key in sorted_si:
        if key in anchor_index.doc_name:
            res.append((key, anchor_index.doc_name[key]))
    # END SOLUTION
    return jsonify(res[:80])


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    page_rank_dictionary = pr.to_dict()
    notExist = 0

    for page_id in wiki_ids:
        if page_id in page_rank_dictionary:
            res.append(page_rank_dictionary[page_id])
        else:
            res.append(notExist)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for pageID in wiki_ids:
        try:
            res.append(dict_page_view[pageID])
        except:
            break
    # END SOLUTION
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)


# -------------->>> Help functions <<<--------------
def merge_results(body_scores, title_scores, anchor_scores, body_weight=0.5, title_weight=0.5, anchor_weight=0.2):
    res = {}

    # body score
    for doc_id, score in body_scores.items():
        if doc_id in res:
            res[doc_id] += score * body_weight
        else:
            res[doc_id] = score * body_weight

    # title score
    for doc_id, score in title_scores.items():
        if doc_id in res:
            res[doc_id] += score * title_weight
        else:
            res[doc_id] = score * title_weight

    # anchor score
    for doc_id, score in anchor_scores.items():
        if doc_id in res:
            res[doc_id] += score * anchor_weight
        else:
            res[doc_id] = score * anchor_weight

    sorted_scores = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return sorted_scores


def top_N_documents(df, N):
    """
    This function sort and filter the top N docuemnts (by score) for each query.

    Parameters
    ----------
    df: DataFrame (queries as rows, documents as columns)
    N: Integer (how many document to retrieve for each query)

    Returns:
    ----------
    top_N: dictionary is the following stracture:
          key - query id.
          value - sorted (according to score) list of pairs lengh of N. Eac pair within the list provide the following information (doc id, score)
    """
    # YOUR CODE HERE
    df_list = df.tolist()
    docs_list = []
    top_N = {}
    for i in range(len(df_list)):
        q = df_list[i]
        temp = []
        for j in range(len(q)):
            temp.append((j, q[j]))
        docs_list.append(temp)
    q_id = 0
    for q in docs_list:
        top_N[q_id] = sorted(q, key=lambda x: x[1], reverse=True)[:N]
        top_N[q_id] = top_N[q_id][:N]
        q_id += 1
    return top_N