"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
from utils.scorer import *


def testing(trainer, opt, test_dataset):
    test_dataset_len = len(test_dataset)
    #batch_size = opt["batch_size"]
    minibatch_size = 1
    a, b, c, d = zip(*test_dataset)
    trainer.eval()
    all_loss = 0
    pre5 = []
    ap5 = []
    ndcg5 = []
    pre7 = []
    ap7 = []
    ndcg7 = []
    pre10 = []
    ap10 = []
    ndcg10 = []
    for i in range(test_dataset_len):
        try:
            supp_xs = list(a[minibatch_size * i:minibatch_size * (i + 1)])
            supp_ys = list(b[minibatch_size * i:minibatch_size * (i + 1)])
            query_xs = list(c[minibatch_size * i:minibatch_size * (i + 1)])
            query_ys = list(d[minibatch_size * i:minibatch_size * (i + 1)])
        except IndexError:
            continue
        test_loss, recommendation_list = trainer.query_rec(supp_xs, supp_ys, query_xs, query_ys)
        all_loss += test_loss

        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre5, ap5, ndcg5, 5)
        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre7, ap7, ndcg7, 7)
        add_metric(recommendation_list, query_ys[0].cpu().detach().numpy(), pre10, ap10, ndcg10, 10)

    mpre5, mndcg5, map5 = cal_metric(pre5, ap5, ndcg5)
    mpre7, mndcg7, map7 = cal_metric(pre7, ap7, ndcg7)
    mpre10, mndcg10, map10 = cal_metric(pre10, ap10, ndcg10)

    return mpre5, mndcg5, map5, mpre7, mndcg7, map7, mpre10, mndcg10, map10
