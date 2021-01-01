"""
Revised Implicit Alternating Least Squares
Reference - https://github.com/benfred/implicit
Only GPU version available
"""
import functools
import heapq
import logging
import time
import torch
import os

import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from tqdm.auto import tqdm

import implicit.cuda

from implicit.als import _als
from implicit.evaluation import train_test_split
from implicit.recommender_base import MatrixFactorizationBase
from implicit.utils import check_blas_config, nonzeros

from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

log = logging.getLogger("implicit")

class Ratings:
    """
    train, test(또는 val) set을 받아서 tag2id dict를 구성하고 ALS 학습을 위한 coo matrix를 생성한다.
    """

    def __init__(self, train, test):  # json, pandas, pandas
        self.train = train
        self.test = test
        self.data = self.train + self.test
        self.num_song = 707989

        self._get_tag2id()

    def _get_tag2id(self):
        """
        tag의 id는 0부터 시작한다. 단 coo matrix를 구성할 때는 0 + self.num_song 부터 시작한다.
        """
        tag_set = set(chain.from_iterable(ply['tags'] for ply in self.data))
        self.num_tag = len(tag_set)
        self.tag2id = {x: i for i, x in enumerate(sorted(tag_set))}
        self.id2tag = {i: x for x, i in self.tag2id.items()}

    def get_raw_tag(self, tids):
        return [self.id2tag[tid] for tid in tids]

    def build_coo(self):
        """
        user id와 item id가 연속적이지 않다면 0인 row가 포함된다.
        ratings의 크기는 (max(uid)+1, max(iid)+1)이 된다.
        """
        pids = []
        iids = []

        for ply in self.data:
            rep = len(ply['songs']) + len(ply['tags'])
            iids.extend(ply['songs'])
            iids.extend(self.tag2id[t] + self.num_song for t in
                        ply['tags'])  # tag2id는 0부터 시작하므로 self.tag2id[t] + self.num_song 임에 주의
            pids.extend([ply['id']] * rep)

        scores = [1] * len(pids)

        ratings = csr_matrix((np.array(scores, dtype=np.float32),
                              (np.array(pids),
                               np.array(iids))),
                             shape=(max(pids) + 1, self.num_song + self.num_tag))

        return ratings

class MyAlternatingLeastSquares(MatrixFactorizationBase):
    """
    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_native : bool, optional
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    use_gpu : bool, optional
        Fit on the GPU if available, default is to run on GPU only if available
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    validate_step : int, optional
        if validate_step > 0, periodically (per validate_step) validates
        the model. validation dataset must be given to argument of
        `fit`method.
        if validate_step <= 0, no validation is done.
    validate_N : int, optional
        size of truncation of validation metric.
        it has no meaning when validate_step <= 0.
    validate_proportion : float, optional
        the portion of validation matrix to total matrix. default value
        to be 0.05
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(self, num_song=707989, num_tag=30197, factors=100,
                 regularization=0.01, dtype=np.float32, use_native=True,
                 use_cg=True, use_gpu=implicit.cuda.HAS_CUDA,
                 iterations=15, calculate_training_loss=False,
                 validate_step=-1, validate_N=30, validate_proportion=0.05,
                 num_threads=0, test_fname="./res/test.json"):
        super(MyAlternatingLeastSquares, self).__init__()
        # currently there are some issues when training on the GPU when some of the warps
        # don't have full factors. Round up to be warp aligned.
        # TODO: figure out where the issue is (best guess is in the
        # the 'dot' function in 'implicit/cuda/utils/cuh)
        if use_gpu and factors % 32:
            padding = 32 - factors % 32
            log.warning("GPU training requires factor size to be a multiple of 32."
                        " Increasing factors from %i to %i.", factors, factors + padding)
            factors += padding

        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization
        self.num_song = num_song
        self.num_tag = num_tag
        self.test_fname = test_fname

        # options on how to fit the model
        self.dtype = dtype
        self.use_native = use_native
        self.use_cg = use_cg
        self.use_gpu = use_gpu
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.num_threads = num_threads
        self.fit_callback = None
        self.cg_steps = 3
        if validate_step < 0:
            self.use_validation = False
        else:
            self.use_validation = True
        self.validate_step = validate_step
        self.validate_N = validate_N
        self.validate_proportion = validate_proportion

        # cache for item factors squared
        self._YtY = None

        check_blas_config()

    def fit(self, item_users, show_progress=True):
        """ Factorizes the item_users matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The item_users matrix does double duty here. It defines which items are liked by which
        users (P_iu in the original paper), as well as how much confidence we have that the user
        liked the item (C_iu).

        The negative items are implicitly defined: This code assumes that non-zero items in the
        item_users matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.

        Parameters
        ----------
        item_users: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the item, the columns are the users that liked that item,
            and the value is the confidence that the user liked the item.
        vali_item_users: csr_matrix
            Same format with item_users. It is used to validate the model.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        """

        #########################################################################################

        if self.use_validation is True:
            item_users, vali_item_users = train_test_split(
                item_users, 1.0 - self.validate_proportion)
            vali_user_items = vali_item_users.T
        else:
            vali_user_items = None

        #########################################################################################

        Ciu = item_users
        if not isinstance(Ciu, scipy.sparse.csr_matrix):
            s = time.time()
            log.debug("Converting input to CSR format")
            Ciu = Ciu.tocsr()
            log.debug("Converted input to CSR in %.3fs", time.time() - s)

        if Ciu.dtype != np.float32:
            Ciu = Ciu.astype(np.float32)

        s = time.time()
        Cui = Ciu.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()
        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = np.random.rand(users, self.factors).astype(self.dtype) * 0.01
        if self.item_factors is None:
            self.item_factors = np.random.rand(items, self.factors).astype(self.dtype) * 0.01

        log.debug("Initialized factors in %s", time.time() - s)

        # invalidate cached norms and squared factors
        self._item_norms = None
        self._YtY = None

        if self.use_gpu:
            print('Use GPU ... Start training')
            return self._fit_gpu(Ciu, Cui, vali_user_items, show_progress)

        raise NotImplementedError("Can't train on CPU faster than original ALS")

    def _fit_gpu(self, Ciu_host, Cui_host, vali_user_items, show_progress=True):
        """ specialized training on the gpu. copies inputs to/from cuda device """
        if not implicit.cuda.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        if self.dtype == np.float64:
            log.warning("Factors of dtype float64 aren't supported with gpu fitting. "
                        "Converting factors to float32")
            self.item_factors = self.item_factors.astype(np.float32)
            self.user_factors = self.user_factors.astype(np.float32)

        Ciu = implicit.cuda.CuCSRMatrix(Ciu_host)
        Cui = implicit.cuda.CuCSRMatrix(Cui_host)
        X = implicit.cuda.CuDenseMatrix(self.user_factors.astype(np.float32))
        Y = implicit.cuda.CuDenseMatrix(self.item_factors.astype(np.float32))

        solver = implicit.cuda.CuLeastSquaresSolver(self.factors)
        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                s = time.time()
                solver.least_squares(Cui, X, Y, self.regularization, self.cg_steps)
                solver.least_squares(Ciu, Y, X, self.regularization, self.cg_steps)
                progress.update(1)

                if self.calculate_training_loss:
                    loss = solver.calculate_loss(Cui, X, Y, self.regularization)
                    progress.set_postfix({"loss": loss})

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

                if self.use_validation and ((iteration + 1) % self.validate_step) == 0:
                    vali_res = self.validate(Cui, vali_user_items, self.validate_N)
                    log.info(
                        "[iter %d] Precision %0.4f MAP %0.4f NDCG %0.4f AUC %0.4f" %
                        (iteration,
                         vali_res["precision"],
                         vali_res["map"],
                         vali_res["ndcg"],
                         vali_res["auc"]))

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

        X.to_host(self.user_factors)  # X를 self.user_factors로 복사
        Y.to_host(self.item_factors)  # Y를 self.item_factors로 복사

        ####################################################################################################

        # song-factor와 tag-factor로 분리해서 저장
        self.y1 = torch.FloatTensor(self.item_factors.T[:, :self.num_song]).to("cuda")
        self.y2 = torch.FloatTensor(self.item_factors.T[:, self.num_song:self.num_song + self.num_tag]).to("cuda")

    ####################################################################################################

    def inference_generator(self):
        """
        self.test_fname에 들어있는 playlist 중 song이나 tag가 있는 playlist에 대해 추천 결과를 return합니다.

        :return: playlist id, rec_songs, rec_tags
        """
        mask = []
        test = pd.read_json(self.test_fname, encoding='UTF-8')
        for i, q in test.iterrows():
            if (q['songs'] != [] or q['tags'] != []):
                mask.append(q['id'])
        del test

        x = torch.FloatTensor(self.user_factors[mask, :]).to("cuda")

        for i in range(x.shape[0]):
            song_score = torch.mm(x[i:(i + 1), ], self.y1)
            tag_score = torch.mm(x[i:(i + 1), ], self.y2)
            indices_song = song_score.argsort(descending=True)
            indices_tag = tag_score.argsort(descending=True)

            yield mask[i], indices_song[0, :200].cpu().tolist(), indices_tag[0, :20].cpu().tolist()

    def recommend(self, pid):
        """
        playlist id를 받아 추천 결과를 return합니다.
        (song, tag가 없는 경우에도 결과를 받을 수 있지만 적절하지 않은 결과가 추출됩니다.)

        :param pid: playlist id
        :return: rec_songs, rec_tags

        """
        x = torch.FloatTensor(self.user_factors[[pid]]).to("cuda")

        song_score = torch.mm(x, self.y1)
        tag_score = torch.mm(x, self.y2)
        indices_song = song_score.argsort(descending=True)
        indices_tag = tag_score.argsort(descending=True)

        return indices_song[0, :200].cpu().tolist(), indices_tag[0, :20].cpu().tolist()

    ####################################################################################################

    def recalculate_user(self, userid, user_items):
        return user_factor(self.item_factors, self.YtY,
                           user_items.tocsr(), userid,
                           self.regularization, self.factors)

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):

        # user_weights = Cholesky decomposition of Wu^-1
        # from section 5 of the paper CF for Implicit Feedback Datasets
        user_items = user_items.tocsr()
        if user_weights is None:
            A, _ = user_linear_equation(self.item_factors, self.YtY,
                                        user_items, userid,
                                        self.regularization, self.factors)
            user_weights = scipy.linalg.cho_factor(A)
        seed_item = self.item_factors[itemid]

        # weighted_item = y_i^t W_u
        weighted_item = scipy.linalg.cho_solve(user_weights, seed_item)

        total_score = 0.0
        h = []
        for i, (itemid, confidence) in enumerate(nonzeros(user_items, userid)):
            if confidence < 0:
                continue

            factor = self.item_factors[itemid]
            # s_u^ij = (y_i^t W^u) y_j
            score = weighted_item.dot(factor) * confidence
            total_score += score
            contribution = (score, itemid)
            if i < N:
                heapq.heappush(h, contribution)
            else:
                heapq.heappushpop(h, contribution)

        items = (heapq.heappop(h) for i in range(len(h)))
        top_contributions = list((i, s) for s, i in items)[::-1]
        return total_score, top_contributions, user_weights

    @property
    def solver(self):
        if self.use_cg:
            solver = _als.least_squares_cg if self.use_native else least_squares_cg
            return functools.partial(solver, cg_steps=self.cg_steps)
        return _als.least_squares if self.use_native else least_squares

    @property
    def YtY(self):
        if self._YtY is None:
            Y = self.item_factors
            self._YtY = Y.T.dot(Y)
        return self._YtY

if __name__ == '__main__':
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    default_dir = './'

    rating_builder = Ratings(train, val_que)
    ratings = rating_builder.build_coo()
    ratings_weighted = (bm25_weight(ratings, B=0.9) * 5).tocsr()

    model = MyAlternatingLeastSquares(num_song=707989, num_tag=30197, factors=128,
                                      regularization=0.01, dtype=np.float32, use_native=True,
                                      use_cg=True, use_gpu=implicit.cuda.HAS_CUDA,
                                      iterations=15, calculate_training_loss=False,
                                      validate_step=-1, validate_N=30, validate_proportion=0.05,
                                      num_threads=0, test_fname=os.path.join(default_dir, 'arena_data',
                                                                             'questions/val_questions.json'))
    model.fit(ratings_weighted.T)
