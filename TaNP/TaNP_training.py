import os
import torch
import pickle
import random
from eval import testing

def training(trainer, opt, train_dataset, test_dataset, batch_size, num_epoch, model_save=True, model_filename=None, logger=None):
    training_set_size = len(train_dataset)
    for epoch in range(num_epoch):
        random.shuffle(train_dataset)
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*train_dataset)
        trainer.train()
        all_C_distribs = []
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            train_loss, batch_C_distribs = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys)
            all_C_distribs.append(batch_C_distribs)

        P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10 = testing(trainer, opt, test_dataset)
        logger.log(
            "{}\t{:.6f}\t TOP-5 {:.4f}\t{:.4f}\t{:.4f}\t TOP-7: {:.4f}\t{:.4f}\t{:.4f}"
            "\t TOP-10: {:.4f}\t{:.4f}\t{:.4f}".
                format(epoch, train_loss, P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10))
        if epoch == (num_epoch-1):
            with open('output_att', 'wb') as fp:
                pickle.dump(all_C_distribs, fp)

    if model_save:
        torch.save(trainer.state_dict(), model_filename)
