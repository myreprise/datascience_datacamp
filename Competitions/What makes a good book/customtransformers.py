from sklearn.base import BaseEstimator, TransformerMixin
import textcleaner as tc
import numpy as np
from nltk.corpus import opinion_lexicon
from sklearn.feature_extraction.text import CountVectorizer


# Custom transformers
class RemapTarget(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy['popularity'] = (X_copy['popularity'] == 'Popular').astype(int)
        return X_copy


class RecodeCategory(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.category_counts = X['categories'].value_counts()
        self.category_mean = self.category_counts.mean()
        self.mapping = self.category_counts[self.category_counts > self.category_mean]
        self.keep_vals = [v.lower() for v in self.mapping.index]
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy['categories'] = X_copy['categories'].str.lower()
        X_copy['categories'] = X_copy['categories'].apply(lambda x: x if x in self.keep_vals else 'other')
        # X_copy['categories'] = np.where(
        #     X_copy['categories'].map(self.category_counts) < self.category_mean,
        #     'other',
        #     X_copy['categories']
        # )
        return X_copy



class SplitReviewCount(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy['review_count'] = X_copy['review/helpfulness'].str.split('/', expand=True)[1].astype(int)
        X_copy['review_helpful'] = X_copy['review/helpfulness'].str.split('/', expand=True)[0].astype(int)

        X_copy['perc_helpful'] = X_copy['review_helpful'] / X_copy['review_count']
        X_copy['perc_helpful'] = X_copy['perc_helpful'].fillna(0)

        X_copy = X_copy.drop(['review/helpfulness', 'review_helpful'], axis = 1)
        return X_copy


class CleanText(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].apply(tc.clean_text)
        return X_copy


class VectorizeCount(BaseEstimator, TransformerMixin):

    def __init__(self, sentiment):
        self.sentiment = sentiment

    def fit(self, X, y=None):
        if self.sentiment == 'positive':
            self.words = [w.lower() for w in opinion_lexicon.positive()]
        else:
            self.words = [w.lower() for w in opinion_lexicon.negative()]        
        return self

    def get_feature_names_out(self):
        return self.get_feature_names()

    def transform(self, X, y = None):
        X_copy = X.copy()
        vectorizer = CountVectorizer(vocabulary=self.words, lowercase = True)
        for col in X_copy.columns:
            transformed = vectorizer.fit_transform(X_copy[col])
            X_copy[col] = transformed.sum(axis = 1).reshape(-1, 1)

        return X_copy