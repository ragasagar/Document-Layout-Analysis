# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import numpy as np
from scipy import spatial
from tqdm import tqdm
from utils.submodlib_util import FL1CMI_wrapper, Margin_Sampling, Random_wrapper, subset
import sys
sys.path.append('../')


class Evaluation(object):

    def make_samples(self):
        raise NotImplementedError("Needs to implemented this method")


def distance(v1, v2, d_type='d1'):
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"

    if d_type == 'd1':
        return np.sum(np.absolute(v1 - v2))
    elif d_type == 'd2':
        return np.sum((v1 - v2) ** 2)
    elif d_type == 'd2-norm':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd3':
        pass
    elif d_type == 'd4':
        pass
    elif d_type == 'd5':
        pass
    elif d_type == 'd6':
        pass
    elif d_type == 'd7':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd8':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'cosine':
        return spatial.distance.cosine(v1, v2)
    elif d_type == 'square':
        return np.sum((v1 - v2) ** 2)


def AP(label, results, sort=True):
    ''' infer a query, return it's ap

      arguments
        label  : query's class
        results: a dict with two keys, see the example below
                 {
                   'dis': <distance between sample & query>,
                   'cls': <sample's class>
                 }
        sort   : sort the results by distance
    '''
    if sort:
        results = sorted(results, key=lambda x: x['dis'])
    precision = []
    hit = 0
    for i, result in enumerate(results):
        if result['cls'] == label:
            hit += 1
            precision.append(hit / (i+1.))
    if hit == 0:
        return 0.
    return np.mean(precision)


def infer_subset(queries, samples=None, db=None, sample_db_fn=None, depth=None,
                 d_type='d1', budget=10, strategy="gcmi", clazz='list', private=None):
    assert samples != None or (db != None and sample_db_fn !=
                               None), "need to give either samples or db plus sample_db_fn"
    if db:
        samples = sample_db_fn(db)

    query_embedding = []

    for idx, query in enumerate(queries, start=1):
        q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
        if q_cls in clazz:
            query_embedding.append(q_hist)

    lake_embedding = []
    lake_image_list = []
    for idx, sample in enumerate(samples, start=1):
        s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
        lake_embedding.append(s_hist)
        lake_image_list.append(s_img)

    if private is not None:
        private_embedding=[]
        for idx, query in enumerate(queries, start=1):
            q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
            if q_cls in private:
                private_embedding.append(q_hist)
        
        subset_result = FL1CMI_wrapper(torch.Tensor(lake_embedding), torch.Tensor(query_embedding), torch.Tensor(private_embedding), 2, lake_image_list, budget=budget, metric='cosine',
                           stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    else:
        subset_result = subset(torch.Tensor(lake_embedding), torch.Tensor(query_embedding), 2, lake_image_list, budget=budget, metric='cosine',
                           stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, strategry=strategy)
    
    return 0, subset_result


def infer(query, samples=None, db=None, sample_db_fn=None, depth=None, d_type='d1'):
    ''' infer a query, return it's ap

      arguments
        query       : a dict with three keys, see the template
                      {
                        'img': <path_to_img>,
                        'cls': <img class>,
                        'hist' <img histogram>
                      }
        samples     : a list of {
                                  'img': <path_to_img>,
                                  'cls': <img class>,
                                  'hist' <img histogram>
                                }
        db          : an instance of class Database
        sample_db_fn: a function making samples, should be given if Database != None
        depth       : retrieved depth during inference, the default depth is equal to database size
        d_type      : distance type
    '''
    assert samples != None or (db != None and sample_db_fn !=
                               None), "need to give either samples or db plus sample_db_fn"
    if db:
        samples = sample_db_fn(db)

    q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
    results = []
    for idx, sample in enumerate(samples):
        s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
        if q_img == s_img:
            continue
        results.append({
            'dis': distance(q_hist, s_hist, d_type=d_type),
            'cls': s_cls
        })
    results = sorted(results, key=lambda x: x['dis'])
    if depth and depth <= len(results):
        results = results[:depth]
    ap = AP(q_cls, results, sort=False)

    return ap, results


def evaluate(db, sample_db_fn, depth=None, d_type='d1'):
    ''' infer the whole database

      arguments
        db          : an instance of class Database
        sample_db_fn: a function making samples, should be given if Database != None
        depth       : retrieved depth during inference, the default depth is equal to database size
        d_type      : distance type
    '''
    classes = db.get_class()
    ret = {c: [] for c in classes}

    samples = sample_db_fn(db)
    for query in samples:
        ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
        ret[query['cls']].append(ap)

    return ret


def evaluate_class(db, f_class=None, f_instance=None, depth=None, d_type='d1'):
    ''' infer the whole database

      arguments
        db     : an instance of class Database
        f_class: a class that generate features, needs to implement make_samples method
        depth  : retrieved depth during inference, the default depth is equal to database size
        d_type : distance type
    '''
    assert f_class or f_instance, "needs to give class_name or an instance of class"

    classes = db.get_class()
    ret = {c: [] for c in classes}

    if f_class:
        f = f_class()
    elif f_instance:
        f = f_instance
    samples = f.make_samples(db)
    for query in tqdm(samples):
        ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
        ret[query['cls']].append(ap)

    return ret
