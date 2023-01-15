from collections import defaultdict
from pathlib import Path
import pickle
import inverted_index_body_gcp 
import inverted_index_title_gcp 
import inverted_index_anchor_gcp 

from contextlib import closing
import math

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

#----read_posting_list----
def read_posting_list(inverted, w, kind):
    if kind == "body":
      with closing(inverted_index_body_gcp.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        #locs_with_path = [(path + t[0], t[1]) for t in locs]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,kind)
        posting_list = []
        for i in range(inverted.df[w]):
          doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
          tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
        return posting_list
    if kind == "title":
      with closing(inverted_index_title_gcp.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        #locs_with_path = [(path + t[0], t[1]) for t in locs]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,kind)
        posting_list = []
        for i in range(inverted.df[w]):
          doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
          tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
        return posting_list
    if kind == "anchor":
      with closing(inverted_index_anchor_gcp.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        #locs_with_path = [(path + t[0], t[1]) for t in locs]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,kind)
        posting_list = []
        for i in range(inverted.df[w]):
          doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
          tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
        return posting_list


class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    docs_len : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avgdl_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    
    kind : type of inverted index body/anchor/title
    """

    def __init__(self, inverted, query,docs_len,kind, k1=1.5, b=0.75):

        self.query = query
        self.inverted = inverted
        self.b = b
        self.k1 = k1
        self.nf = docs_len
        self.N_ = len(docs_len)
        self.avgdl_ = sum(docs_len.values()) / len(docs_len)
        self.candidates_tf = defaultdict(list)
        self.idf_dic = defaultdict(list)
        self.kind = kind


    def calc_idf(self):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """        

        for word in self.query:
            # if the term already added
            if word in self.idf_dic:
                continue
            else:
                # if the term exist in the corpus
                self.idf_dic[word] = math.log10((self.N_ / (self.inverted.df[word] + 0.5)) + 1)

        return self.idf_dic

    def doc_score(self, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0

        for word in self.query:
            curr_tf = ""
            if doc_id in self.candidates_tf:
                for tup in self.candidates_tf[doc_id]: #look for the tuple (word, term frequency) in the list for the current document id
                    if tup[0] == word:
                        curr_tf = tup[1] #set the current term frequency (curr_tf) to the term frequency value
                        continue
            if word in self.idf_dic and doc_id in self.nf: 
                if curr_tf != "":
                    #calculate BM25 score
                    score += ((self.idf_dic[word]) * curr_tf * (self.k1 + 1)) / (
                            curr_tf + self.k1 * (
                            1 - self.b + ((self.b * self.nf[doc_id]) / self.avgdl_)))
        return score

    def create_tf(self):
        """
        This function creates a dictionary of candidates and their term frequency.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        update the candidates_tf
        """
        c = {}
        for word in self.query:
            words_posting_lst = read_posting_list(self.inverted, word, self.kind)
            for tup in words_posting_lst:
                c[tup[0]] = []
                c[tup[0]].append((word, tup[1]))# Append the tuple (word, term frequency) to the list for the current document id
        self.candidates_tf = c

    def calc_candidates_score(self):
        """
        This function calculates the BM25 score for each candidate.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        return the top 100 candidates from the sorted scores dictionary
        """
        
        scores = {}
        self.create_tf()  # call the create_tf function to create the candidates_tf dictionary
        self.calc_idf()  # call the calc_idf function to calculate the idf values
        for c in self.candidates_tf:
            scores[c] = self.doc_score(c) # for each candidate, calculate the score using the doc_score function and store it in the scores dictionary with the document id as the key
        res = dict(sorted(scores.items(), key=lambda item: -item[1]))
        return dict(list(res.items())[:100])
