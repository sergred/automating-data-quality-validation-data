import warnings
warnings.filterwarnings('ignore')

from collections import namedtuple, Counter as count
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from hyperloglog import HyperLogLog
from pyod.models.knn import KNN
from abc import abstractmethod
from dabl import detect_types
from nltk.util import ngrams
from pathlib import Path
from enum import IntEnum
import pandas as pd
import numpy as np
import copy
import time


np.random.seed(42)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r took %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


class Quality(IntEnum):
    GOOD = 0
    BAD = 1


class Learner:
    @abstractmethod
    def fit(history):
        pass

    @abstractmethod
    def predict(X):
        pass


class KNNLearner(Learner):
    def __init__(self):
        # https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.knn
        self.clf = None

    def fit(self, history):
        learner = KNN(contamination=.01,
                      n_neighbors=5,
                      method='mean',
                      metric='euclidean',
                      algorithm='ball_tree')
        self.clf = Pipeline([
            ('scaler', MinMaxScaler()),
            ('learner', learner)
        ]).fit(history)

        return self

    def predict(self, X):
        assert self.clf is not None, ".fit first"
        return self.clf.predict(X)


class DataProfiler:
    class __DP:
        def __init__(self):
            self.analyzer = {
                "Completeness": lambda x: self.completeness(x),
                "Uniqueness": lambda x: self.uniqueness(x),
                "ApproxCountDistinct": lambda x: self.approx_count_distinct(x),
                "Mean": lambda x: np.mean(x),
                "Minimum": lambda x: np.min(x),
                "Maximum": lambda x: np.max(x),
                "StandardDeviation": lambda x: np.std(x),
                "Sum": lambda x: np.sum(x),
                "Count": lambda x: x.shape[0],
                "FrequentRatio": lambda x: 1.*max(count(x).values())/x.shape[0],
                "PeculiarityIndex": lambda x: self.peculiarity(x),
            }

            self.dtype_checking = {
                "int64": True,
                "float64": True
            }


        def completeness(self, x):
            return 1. - np.sum(pd.isna(x)) / x.shape[0]

        def uniqueness(self, x):
            tmp = [i for i in count(x).values() if i == 1]
            return 1. * np.sum(tmp) / x.shape[0]

        def count_distinct(self, x):
            return 1. * len(count(x).keys()) / x.shape[0]

        def approx_count_distinct(self, x):
            hll = HyperLogLog(.01)
            for idx, val in x.items():
                hll.add(str(val))
            return len(hll)

        # TODO: count sketch, using deterministic count for small data
#        def count_sketch(self, matrix, sketch_size=50):
#            m, n = matrix.shape[0], 1
#            res = np.zeros([m, sketch_size])
#            hashedIndices = np.random.choice(sketch_size, replace=True)
#            print(hashedIndices)
#            randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
#            matrix = matrix * randSigns
#            for i in range(sketch_size):
#                res[:, i] = np.sum(matrix[:, hashedIndices == i], 1)
#            return res

        def peculiarity(self, x):
            def _peculiarity_index(word, count2grams, count3grams):
                t = []
                for xyz in ngrams(str(word), 3):
                    xy, yz = xyz[:2], xyz[1:]
                    cxy, cyz = count2grams.get(xy, 0), count2grams.get(yz, 0)
                    cxyz = count3grams.get(xyz, 0)
                    t.append(.5* (np.log(cxy) + np.log(cyz) - np.log(cxyz)))
                return np.sqrt(np.mean(np.array(t)**2))

            aggregated_string = " ".join(map(str, x))
            c2gr = count(ngrams(aggregated_string, 2))
            c3gr = count(ngrams(aggregated_string, 3))
            return x.apply(lambda y: _peculiarity_index(y, c2gr, c3gr)).max()

    instance = None

    def __init__(self):
        if not DataProfiler.instance:
            DataProfiler.instance = DataProfiler.__DP()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def _compute_for_column(self, column, *analyzers):
        return [self.instance.analyzer[name](column) for name in analyzers]

    # @timeit
    def compute_for(self, batch, return_labels=False):
        profile, labels = [], []
        generic_metrics = ["Completeness", "Uniqueness",
                           "ApproxCountDistinct", "FrequentRatio"]
        numeric_metrics = ["Mean", "Minimum", "Maximum",
                           "StandardDeviation", "Sum"]

        is_free_string = detect_types(batch)['free_string']
        for col, dtype in zip(batch.columns, batch.dtypes):
            # For every column, compute generic metrics,
            # add additional numeric metrics for numeric columns
            metrics = copy.deepcopy(generic_metrics)
            if self.dtype_checking.get(dtype, False):
                metrics.extend(numeric_metrics)
            if dtype == 'object': # Dummy check for likely-to-be-strings
                metrics.append("PeculiarityIndex")
            # print(col, dtype, metrics)
            # We assume the data schema to be stable, column order unchanged,
            # no additional validation for feature order happens, optional
            column_profile = self._compute_for_column(batch[col], *metrics)
            profile.extend(column_profile)
            labels.extend([f'{col}_{m}' for m in metrics])
        return profile if not return_labels else (profile, labels)


class DataQualityValidatior:
    def __init__(self):
        self.clf = KNNLearner()
        self.history = []

    def add(self, batch):
        self.history.append(batch)
        return self

    def test(self, batch):
        # print(len(self.history))
        # re-fit the model from scratch
        self.clf.fit(self.history)

        decision = self.clf.predict([batch])
        return Quality.GOOD if int(decision) == 0 else Quality.BAD


Batch = namedtuple('Batch', 'id clean dirty')

def get_batch_fnames():
    folder = Path('partitions/')
    batches = []
    for day in range(1, 54):
        fclean = folder / f'clean/FBPosts_clean_{day}.tsv'
        fdirty = folder / f'dirty/FBPosts_dirty_{day}.tsv'
        assert fclean.exists()
        assert fdirty.exists()
        batches.append(Batch(day, fclean, fdirty))
    return iter(batches)


def good_or_bad(batch):
    return np.random.choice([Quality.GOOD, Quality.BAD], p=[9./10, 1./10])


def demo():
    dqv = DataQualityValidatior()
    batches = get_batch_fnames()
    # initial training set
    for day in range(8):
        batch = next(batches)
        # TODO: provide functionality to read the data
        # in the FBPosts case, .tsv files
        batch_data = pd.read_csv(batch.clean, sep='\t')
        profile = DataProfiler().compute_for(batch_data)
        dqv.add(profile)

    # testing phase
    for day in range(20):
        batch = next(batches)
        chance = good_or_bad(batch)
        print("Good" if chance == Quality.GOOD else " Bad", f'batch {batch.id} is coming.')
        fname = batch.clean if chance == Quality.GOOD else batch.dirty
        batch_data = pd.read_csv(fname, sep='\t')
        profile = DataProfiler().compute_for(batch_data)

        res = dqv.test(profile)
        if res == Quality.GOOD:
            dqv.add(profile)
        else:
            # TODO: Alarm, debugging, fixing, pass if false alarm
            print(f'**** Potential problem with batch #{batch.id}!')
            fixed_data = pd.read_csv(batch.clean, sep='\t')
            fixed_profile = DataProfiler().compute_for(fixed_data)
            dqv.add(fixed_profile)


if __name__ == "__main__":
    demo()

