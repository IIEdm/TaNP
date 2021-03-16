import json
import random
import torch
import numpy as np
import pickle
import codecs
import re
import os
import datetime
import tqdm
import pandas as pd

#convert userids to userdict key-id(int), val:onehot_vector(tensor)
#element in list is str type.
def to_onehot_dict(list):
    dict={}
    length = len(list)
    for index, element in enumerate(list):
        vector = torch.zeros(1, length).long()
        element = int(element)
        vector[:, element] = 1.0
        dict[element] = vector
    return dict

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

# used for merge dictionaries.
def merge_key(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def merge_value(dict1, dict2): # merge and item_cold
    for key, value in dict2.items():
        if key in dict1.keys():
            # if list(set(dict1[key]+value)) the final number of movies-1m is 1000205
            new_value = dict1[key]+value
            dict1[key] = new_value
        else:
            print('Unexpected key.')

def count_values(dict):
    count_val = 0
    for key, value in dict.items():
        count_val += len(value)
    return count_val

def construct_dictionary(user_list, total_dict):
    dict = {}
    for i in range(len(user_list)):
        dict[str(user_list[i])] = total_dict[str(user_list[i])]
    return dict

class Preprocess(object):
    """
    Preprocess the training, validation and test data.
    Generate the episode-style data.
    """

    def __init__(self, opt):
        self.batch_size = opt["batch_size"]
        self.opt = opt
        # warm data ratio
        self.train_ratio = opt['train_ratio']
        self.valid_ratio = opt['valid_ratio']
        self.test_ratio = 1 - self.train_ratio - self.valid_ratio
        self.dataset_path = opt["data_dir"]
        self.support_size = opt['support_size']
        self.query_size = opt['query_size']
        self.max_len = opt['max_len']
        # save one-hot dimension length
        uf_dim, if_dim = self.preprocess(self.dataset_path)
        self.uf_dim = uf_dim
        self.if_dim = if_dim

    def preprocess(self, dataset_path):
        """ Preprocess the data and convert to ids. """
        #Create training-validation-test datasets
        print('Create training, validation and test data from scratch!')
        with open('./{}/interaction_dict_x.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            inter_dict_x = json.loads(f.read())
        with open('./{}/interaction_dict_y.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            inter_dict_y = json.loads(f.read())
        print('The size of total interactions is %d.' % (count_values(inter_dict_x)))  # 42346
        assert count_values(inter_dict_x) == count_values(inter_dict_y)

        with open('./{}/user_list.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            userids = json.loads(f.read())

        with open('./{}/item_list.json'.format(dataset_path), 'r', encoding='utf-8') as f:
            itemids = json.loads(f.read())

        #userids = list(inter_dict_x.keys())
        random.shuffle(userids)
        warm_user_size = int(len(userids) * self.train_ratio)
        valid_user_size = int(len(userids) * self.valid_ratio)
        warm_users = userids[:warm_user_size]
        valid_users = userids[warm_user_size:warm_user_size+valid_user_size]
        cold_users = userids[warm_user_size+valid_user_size:]
        assert len(userids) == len(warm_users)+len(valid_users)+len(cold_users)


            # Construct the training data dict
        training_dict_x = construct_dictionary(warm_users, inter_dict_x)
        training_dict_y = construct_dictionary(warm_users, inter_dict_y)

            #Avoid the new items shown in test data in the case of cold user.
        item_set = set()
        for i in training_dict_x.values():
            i = set(i)
            item_set = item_set.union(i)

        # Construct one-hot dictionary
        user_dict = to_onehot_dict(userids)
        # only items contained in all data are encoded.
        item_dict = to_onehot_dict(itemids)

        # This part of data is not used, so we do not process it temporally.
        valid_dict_x = construct_dictionary(valid_users, inter_dict_x)
        valid_dict_y = construct_dictionary(valid_users, inter_dict_y)
        assert count_values(valid_dict_x) == count_values(valid_dict_y)

        test_dict_x = construct_dictionary(cold_users, inter_dict_x)
        test_dict_y = construct_dictionary(cold_users, inter_dict_y)
        assert count_values(test_dict_x) == count_values(test_dict_y)

        print('Before delete new items in test data, test data has %d interactions.' % (count_values(test_dict_x)))

        #Delete the new items in test data.
        unseen_count = 0
        for key, value in test_dict_x.items():
            assert len(value) == len(test_dict_y[key])
            unseen_item_index = [index for index, i in enumerate(value) if i not in item_set]
            unseen_count+=len(unseen_item_index)
            if len(unseen_item_index) == 0:
                continue
            else:
                new_value_x = [element for index, element in enumerate(value) if index not in unseen_item_index]
                new_value_y = [test_dict_y[key][index] for index, element in enumerate(value) if index not in unseen_item_index]
                test_dict_x[key] = new_value_x
                test_dict_y[key] = new_value_y
        print('After delete new items in test data, test data has %d interactions.' % (count_values(test_dict_x)))
        assert count_values(test_dict_x) == count_values(test_dict_y)
        print('The number of total unseen interactions is %d.' % (unseen_count))

        pickle.dump(training_dict_x, open("{}/training_dict_x_{:2f}.pkl".format(dataset_path, self.train_ratio), "wb"))
        pickle.dump(training_dict_y, open("{}/training_dict_y_{:2f}.pkl".format(dataset_path, self.train_ratio), "wb"))
        pickle.dump(valid_dict_x, open("{}/valid_dict_x_{:2f}.pkl".format(dataset_path, self.valid_ratio), "wb"))
        pickle.dump(valid_dict_y, open("{}/valid_dict_y_{:2f}.pkl".format(dataset_path, self.valid_ratio), "wb"))
        pickle.dump(test_dict_x, open("{}/test_dict_x_{:2f}.pkl".format(dataset_path, self.test_ratio), "wb"))
        pickle.dump(test_dict_y, open("{}/test_dict_y_{:2f}.pkl".format(dataset_path, self.test_ratio), "wb"))

        def generate_episodes(dict_x, dict_y, category, support_size, query_size, max_len, dir="log"):
            idx = 0
            if not os.path.exists("{}/{}/{}".format(dataset_path, category, dir)):
                os.makedirs("{}/{}/{}".format(dataset_path, category, dir))
                os.makedirs("{}/{}/{}".format(dataset_path, category, "evidence"))
                for _, user_id in enumerate(dict_x.keys()):
                    u_id = int(user_id)
                    seen_music_len = len(dict_x[str(u_id)])
                    indices = list(range(seen_music_len))
                    # filter some users with their interactions, i.e., tasks
                    if seen_music_len < (support_size + query_size) or seen_music_len > max_len:
                        continue
                    random.shuffle(indices)
                    tmp_x = np.array(dict_x[str(u_id)])
                    tmp_y = np.array(dict_y[str(u_id)])

                    support_x_app = None
                    for m_id in tmp_x[indices[:support_size]]:
                        m_id = int(m_id)
                        tmp_x_converted = torch.cat((item_dict[m_id], user_dict[u_id]), 1)
                        try:
                            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                        except:
                            support_x_app = tmp_x_converted

                    query_x_app = None
                    for m_id in tmp_x[indices[support_size:]]:
                        m_id = int(m_id)
                        u_id = int(user_id)
                        tmp_x_converted = torch.cat((item_dict[m_id], user_dict[u_id]), 1)
                        try:
                            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                        except:
                            query_x_app = tmp_x_converted

                    support_y_app = torch.FloatTensor(tmp_y[indices[:support_size]])
                    query_y_app = torch.FloatTensor(tmp_y[indices[support_size:]])

                    pickle.dump(support_x_app, open("{}/{}/{}/supp_x_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(support_y_app, open("{}/{}/{}/supp_y_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(query_x_app, open("{}/{}/{}/query_x_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    pickle.dump(query_y_app, open("{}/{}/{}/query_y_{}.pkl".format(dataset_path, category, dir, idx), "wb"))
                    # used for evidence candidate selection
                    with open("{}/{}/{}/supp_x_{}_u_m_ids.txt".format(dataset_path, category, "evidence", idx), "w") as f:
                        for m_id in tmp_x[indices[:support_size]]:
                            f.write("{}\t{}\n".format(u_id, m_id))
                    with open("{}/{}/{}/query_x_{}_u_m_ids.txt".format(dataset_path, category, "evidence", idx), "w") as f:
                        for m_id in tmp_x[indices[support_size:]]:
                            f.write("{}\t{}\n".format(u_id, m_id))
                    idx+=1

        print("Generate eposide data for training.")
        generate_episodes(training_dict_x, training_dict_y, "training", self.support_size, self.query_size, self.max_len)
        print("Generate eposide data for validation.")
        generate_episodes(valid_dict_x, valid_dict_y, "validation", self.support_size, self.query_size, self.max_len)
        print("Generate eposide data for testing.")
        generate_episodes(test_dict_x, test_dict_y, "testing", self.support_size, self.query_size, self.max_len)

        return len(userids), len(itemids)



