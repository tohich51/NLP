from sklearn.base import TransformerMixin
from collections import OrderedDict
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        
        self.bow = OrderedDict()
        
        for t in X:
            for w in t.split(' '):
                try:
                    self.bow[w] += 1
                except:
                    self.bow[w] = 1
                    
        self.bow = list(OrderedDict(sorted(self.bow.items(), key=lambda x: x[1], reverse=True)[:self.k]).keys())
        
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = [0] * self.k
        for i, w in enumerate(text.split()):
            try:
                result[self.bow.index(w)] += 1
            except:
                pass
        
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        self.N = X.shape[0]
            
        bow = BoW(self.k)
        bow.fit(X)
        unique_words = bow.get_vocabulary()

        # IDF
        for w in unique_words:
            for text in X:
                for words in set(text.split()):
                    if w in words:
                        try:
                            self.idf[w] += 1
                        except:
                            self.idf[w] = 0
                try:
                    self.idf[w] = np.log(self.N / (self.idf[w] + 1))
                except:
                    pass
        
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = [0] * len(self.idf)
        vocabulary = list(self.idf.keys())
        words = text.split()
        
        for word in words:
            try:
                result[vocabulary.index(word)] += 1
            except:
                pass
        
        for key, item in self.idf.items():
            result[vocabulary.index(key)] *= item
            
        if self.normalize:
            s = sum(result)
            result = [r / s if s > 0 else 0 for r in result]
            
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
