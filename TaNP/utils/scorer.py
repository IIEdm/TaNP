import math

def AP(ranked_list, ground_truth, topn):
    hits, sum_precs = 0, 0.0
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t=t[:topn]
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            hits += 1
            sum_precs += hits / (i+1.0)
            t.remove(ground_truth[id])
    if hits > 0:
        return sum_precs / topn
    else:
        return 0.0

def RR(ranked_list, ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            return 1 / (i + 1.0)
    return 0

def precision(ranked_list,ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    hits = 0
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            t.remove(ground_truth[id])
            hits += 1
    pre = hits/topn
    return pre


def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    # print(ranked_list)
    # input()
    for i in range(topn):
        id = ranked_list[i]
        dcg += ((2 ** ground_truth[id]) -1)/ math.log(i+2, 2)
    # print('dcg is ', dcg, " n is ", topn)
    # print('idcg is ', idcg, " n is ", topn)
    return dcg / idcg

def IDCG(ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
    return idcg

def add_metric(recommend_list, ALL_group_list, precision_list, ap_list, ndcg_list, topn):
    ndcg = nDCG(recommend_list, ALL_group_list, topn)
    ap = AP(recommend_list, ALL_group_list, topn)
    pre = precision(recommend_list, ALL_group_list, topn)
    precision_list.append(pre)
    ap_list.append(ap)
    ndcg_list.append(ndcg)



def cal_metric(precision_list,ap_list,ndcg_list):
    mpre = sum(precision_list) / len(precision_list)
    map = sum(ap_list) / len(ap_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return mpre, mndcg, map
