import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from utils.loader import Preprocess
from TaNP import Trainer
from TaNP_training import training
from utils import helper
from eval import testing

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/lastfm_20')#1
parser.add_argument('--model_save_dir', type=str, default='save_model_dir')#1
parser.add_argument('--id', type=str, default='1', help='used for save hyper-parameters.')#1

parser.add_argument('--first_embedding_dim', type=int, default=32, help='Embedding dimension for item and user.')#1
parser.add_argument('--second_embedding_dim', type=int, default=16, help='Embedding dimension for item and user.')#1

parser.add_argument('--z1_dim', type=int, default=32, help='The dimension of z1 in latent path.')
parser.add_argument('--z2_dim', type=int, default=32, help='The dimension of z2 in latent path.')
parser.add_argument('--z_dim', type=int, default=32, help='The dimension of z in latent path.')

parser.add_argument('--enc_h1_dim', type=int, default=64, help='The hidden first dimension of encoder.')
parser.add_argument('--enc_h2_dim', type=int, default=64, help='The hidden second dimension of encoder.')

parser.add_argument('--taskenc_h1_dim', type=int, default=128, help='The hidden first dimension of task encoder.')
parser.add_argument('--taskenc_h2_dim', type=int, default=64, help='The hidden second dimension of task encoder.')
parser.add_argument('--taskenc_final_dim', type=int, default=64, help='The hidden second dimension of task encoder.')

parser.add_argument('--clusters_k', type=int, default=7, help='Cluster numbers of tasks.')
parser.add_argument('--temperature', type=float, default=1.0, help='used for student-t distribution.')
parser.add_argument('--lambda', type=float, default=0.1, help='used to balance the clustering loss and NP loss.')

parser.add_argument('--dec_h1_dim', type=int, default=128, help='The hidden first dimension of encoder.')
parser.add_argument('--dec_h2_dim', type=int, default=128, help='The hidden second dimension of encoder.')
parser.add_argument('--dec_h3_dim', type=int, default=128, help='The hidden third dimension of encoder.')

# used for movie datasets
#parser.add_argument('--num_gender', type=int, default=2, help='User information.')#1
#parser.add_argument('--num_age', type=int, default=7, help='User information.')#1
#parser.add_argument('--num_occupation', type=int, default=21, help='User information.')#1
#parser.add_argument('--num_zipcode', type=int, default=3402, help='User information.')#1
#parser.add_argument('--num_rate', type=int, default=6, help='Item information.')#1
#parser.add_argument('--num_genre', type=int, default=25, help='Item information.')#1
#parser.add_argument('--num_director', type=int, default=2186, help='Item information.')#1
#parser.add_argument('--num_actor', type=int, default=8030, help='Item information.')#1

parser.add_argument('--dropout_rate', type=float, default=0, help='used in encoder and decoder.')
parser.add_argument('--lr', type=float, default=1e-4, help='Applies to SGD and Adagrad.')#1
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=150)#1
parser.add_argument('--batch_size', type=int, default=32)#1
parser.add_argument('--train_ratio', type=float, default=0.7, help='Warm user ratio for training.')#1
parser.add_argument('--valid_ratio', type=float, default=0.1, help='Cold user ratio for validation.')#1
parser.add_argument('--seed', type=int, default=2020)#1
parser.add_argument('--save', type=int, default=0)#1
parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available())#1
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')#1
parser.add_argument('--support_size', type=int, default=20)#1
parser.add_argument('--query_size', type=int, default=10)#1
parser.add_argument('--max_len', type=int, default=200, help='The max length of interactions for each user.')
parser.add_argument('--context_min', type=int, default=20, help='Minimum size of context range.')
args = parser.parse_args()

def seed_everything(seed=1023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = args.seed
seed_everything(seed)

if args.cpu:
    args.use_cuda = False
elif args.use_cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

# print model info
helper.print_config(opt)
helper.ensure_dir(opt["model_save_dir"], verbose=True)
# save model config
helper.save_config(opt, opt["model_save_dir"] + "/" +opt["id"] + '.config', verbose=True)
# record training log
file_logger = helper.FileLogger(opt["model_save_dir"] + '/' + opt['id'] + ".log",
                                header="# epoch\ttrain_loss\tprecision5\tNDCG5\tMAP5\tprecision7"
                                       "\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")

preprocess = Preprocess(opt)
print("Preprocess is done.")
print("Create model TaNP...")

opt['uf_dim'] = preprocess.uf_dim
opt['if_dim'] = preprocess.if_dim

trainer = Trainer(opt)
if opt['use_cuda']:
    trainer.cuda()

model_filename = "{}/{}.pt".format(opt['model_save_dir'], opt["id"])

# /4 since sup_x, sup_y, query_x, query_y
training_set_size = int(len(os.listdir("{}/{}/{}".format(opt["data_dir"], "training", "log"))) / 4)
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []
for idx in range(training_set_size):
    supp_xs_s.append(pickle.load(open("{}/{}/{}/supp_x_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    supp_ys_s.append(pickle.load(open("{}/{}/{}/supp_y_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    query_xs_s.append(pickle.load(open("{}/{}/{}/query_x_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
    query_ys_s.append(pickle.load(open("{}/{}/{}/query_y_{}.pkl".format(opt["data_dir"], "training", "log", idx), "rb")))
train_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

testing_set_size = int(len(os.listdir("{}/{}/{}".format(opt["data_dir"], "testing", "log"))) / 4)
supp_xs_s = []
supp_ys_s = []
query_xs_s = []
query_ys_s = []
for idx in range(testing_set_size):
    supp_xs_s.append(
        pickle.load(open("{}/{}/{}/supp_x_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    supp_ys_s.append(
        pickle.load(open("{}/{}/{}/supp_y_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    query_xs_s.append(
        pickle.load(open("{}/{}/{}/query_x_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
    query_ys_s.append(
        pickle.load(open("{}/{}/{}/query_y_{}.pkl".format(opt["data_dir"], "testing", "log", idx), "rb")))
test_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

print("# epoch\ttrain_loss\tprecision5\tNDCG5\tMAP5\tprecision7\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")

if not os.path.exists(model_filename):
    print("Start training...")
    training(trainer, opt, train_dataset, test_dataset, batch_size=opt['batch_size'], num_epoch=opt['num_epoch'],
            model_save=opt["save"], model_filename=model_filename, logger=file_logger)

else:
    print("Load pre-trained model...")
    opt = helper.load_config(model_filename[:-2]+"config")
    helper.print_config(opt)
    trained_state_dict = torch.load(model_filename)
    trainer.load_state_dict(trained_state_dict)

