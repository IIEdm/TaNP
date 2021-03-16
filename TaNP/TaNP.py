import torch
import numpy as np
from random import randint
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from embeddings_TaNP import Item, User, Encoder, MuSigmaEncoder, Decoder, Gating_Decoder, TaskEncoder, MemoryUnit
import torch.nn as nn

class NP(nn.Module):
    def __init__(self, config):
        super(NP, self).__init__()
        self.x_dim = config['second_embedding_dim'] * 2
        # use one-hot or not?
        self.y_dim = 1
        self.z1_dim = config['z1_dim']
        self.z2_dim = config['z2_dim']
        # z is the dimension size of mu and sigma.
        self.z_dim = config['z_dim']
        # the dimension size of rc.
        self.enc_h1_dim = config['enc_h1_dim']
        self.enc_h2_dim = config['enc_h2_dim']

        self.dec_h1_dim = config['dec_h1_dim']
        self.dec_h2_dim = config['dec_h2_dim']
        self.dec_h3_dim = config['dec_h3_dim']

        self.taskenc_h1_dim = config['taskenc_h1_dim']
        self.taskenc_h2_dim = config['taskenc_h2_dim']
        self.taskenc_final_dim = config['taskenc_final_dim']

        self.clusters_k = config['clusters_k']
        self.temperture = config['temperature']
        self.dropout_rate = config['dropout_rate']

        # Initialize networks
        self.item_emb = Item(config)
        self.user_emb = User(config)
        # This encoder is used to generated z actually, it is a latent encoder in ANP.
        self.xy_to_z = Encoder(self.x_dim, self.y_dim, self.enc_h1_dim, self.enc_h2_dim, self.z1_dim, self.dropout_rate)
        self.z_to_mu_sigma = MuSigmaEncoder(self.z1_dim, self.z2_dim, self.z_dim)
        # This encoder is used to generated r actually, it is a deterministic encoder in ANP.
        self.xy_to_task = TaskEncoder(self.x_dim, self.y_dim, self.taskenc_h1_dim, self.taskenc_h2_dim, self.taskenc_final_dim,
                                      self.dropout_rate)
        self.memoryunit = MemoryUnit(self.clusters_k, self.taskenc_final_dim, self.temperture)
        #self.xz_to_y = Gating_Decoder(self.x_dim, self.z_dim, self.taskenc_final_dim, self.dec_h1_dim, self.dec_h2_dim, self.dec_h3_dim, self.y_dim, self.dropout_rate)
        self.xz_to_y = Decoder(self.x_dim, self.z_dim, self.taskenc_final_dim, self.dec_h1_dim, self.dec_h2_dim, self.dec_h3_dim, self.y_dim, self.dropout_rate)

    def aggregate(self, z_i):
        return torch.mean(z_i, dim=0)

    def xy_to_mu_sigma(self, x, y):
        # Encode each point into a representation r_i
        z_i = self.xy_to_z(x, y)
        # Aggregate representations r_i into a single representation r
        z = self.aggregate(z_i)
        # Return parameters of distribution
        return self.z_to_mu_sigma(z)

    # embedding each (item, user) as the x for np
    def embedding(self, x):
        if_dim = self.item_emb.feature_dim
        item_x = Variable(x[:, 0:if_dim], requires_grad=False).float()
        user_x = Variable(x[:, if_dim:], requires_grad=False).float()
        item_emb = self.item_emb(item_x)
        user_emb = self.user_emb(user_x)
        x = torch.cat((item_emb, user_emb), 1)
        return x

    def forward(self, x_context, y_context, x_target, y_target):
        x_context_embed = self.embedding(x_context)
        x_target_embed = self.embedding(x_target)

        if self.training:
            # sigma is log_sigma actually
            mu_target, sigma_target, z_target = self.xy_to_mu_sigma(x_target_embed, y_target)
            mu_context, sigma_context, z_context = self.xy_to_mu_sigma(x_context_embed, y_context)
            task = self.xy_to_task(x_context_embed, y_context)
            mean_task = self.aggregate(task)
            C_distribution, new_task_embed = self.memoryunit(mean_task)
            p_y_pred = self.xz_to_y(x_target_embed, z_target, new_task_embed)
            return p_y_pred, mu_target, sigma_target, mu_context, sigma_context, C_distribution
        else:
            mu_context, sigma_context, z_context = self.xy_to_mu_sigma(x_context_embed, y_context)
            task = self.xy_to_task(x_context_embed, y_context)
            mean_task = self.aggregate(task)
            C_distribution, new_task_embed = self.memoryunit(mean_task)
            p_y_pred = self.xz_to_y(x_target_embed, z_context, new_task_embed)
            return p_y_pred


class Trainer(torch.nn.Module):
    def __init__(self, config):
        self.opt = config
        super(Trainer, self).__init__()
        self.use_cuda = config['use_cuda']
        self.np = NP(self.opt)
        self._lambda = config['lambda']
        self.optimizer = torch.optim.Adam(self.np.parameters(), lr=config['lr'])

    # our kl divergence
    def kl_div(self, mu_target, logsigma_target, mu_context, logsigma_context):
        target_sigma = torch.exp(logsigma_target)
        context_sigma = torch.exp(logsigma_context)
        kl_div = (logsigma_context - logsigma_target) - 0.5 + (((target_sigma ** 2) + (mu_target - mu_context) ** 2) / 2 * context_sigma ** 2)
        #kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        #kl_div = 0.5 * kl_div.sum()
        kl_div = kl_div.sum()
        return kl_div

    # new kl divergence -- kl(st|sc)
    def new_kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

    def loss(self, p_y_pred, y_target, mu_target, sigma_target, mu_context, sigma_context):
        #print('p_y_pred size is ', p_y_pred.size())
        regression_loss = F.mse_loss(p_y_pred, y_target.view(-1, 1))
        #print('regession loss size is ', regression_loss.size())
        # kl divergence between target and context
        #print('regession_loss is ', regression_loss.item())
        kl = self.new_kl_div(mu_context, sigma_context, mu_target, sigma_target)
        #print('KL_loss is ', kl.item())
        return regression_loss+kl

    def context_target_split(self, support_set_x, support_set_y, query_set_x, query_set_y):
        total_x = torch.cat((support_set_x, query_set_x), 0)
        total_y = torch.cat((support_set_y, query_set_y), 0)
        total_size = total_x.size(0)
        context_min = self.opt['context_min']
        context_max = self.opt['context_max']
        extra_tar_min = self.opt['target_extra_min']
        #here we simply use the total_size as the maximum of target size.
        num_context = randint(context_min, context_max)
        num_target = randint(extra_tar_min, total_size - num_context)
        sampled = np.random.choice(total_size, num_context+num_target, replace=False)
        x_context = total_x[sampled[:num_context], :]
        y_context = total_y[sampled[:num_context]]
        x_target = total_x[sampled, :]
        y_target = total_y[sampled]
        return x_context, y_context, x_target, y_target

    def new_context_target_split(self, support_set_x, support_set_y, query_set_x, query_set_y):
        total_x = torch.cat((support_set_x, query_set_x), 0)
        total_y = torch.cat((support_set_y, query_set_y), 0)
        total_size = total_x.size(0)
        context_min = self.opt['context_min']
        num_context = np.random.randint(context_min, total_size)
        num_target = np.random.randint(0, total_size - num_context)
        sampled = np.random.choice(total_size, num_context+num_target, replace=False)
        x_context = total_x[sampled[:num_context], :]
        y_context = total_y[sampled[:num_context]]
        x_target = total_x[sampled, :]
        y_target = total_y[sampled]
        return x_context, y_context, x_target, y_target

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = len(support_set_xs)
        losses = []
        C_distribs = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            x_context, y_context, x_target, y_target = self.new_context_target_split(support_set_xs[i], support_set_ys[i],
                                                                                 query_set_xs[i], query_set_ys[i])
            p_y_pred, mu_target, sigma_target, mu_context, sigma_context, C_distribution = self.np(x_context, y_context, x_target,
                                                                                  y_target)
            C_distribs.append(C_distribution)
            loss = self.loss(p_y_pred, y_target, mu_target, sigma_target, mu_context, sigma_context)
            #print('Each task has loss: ', loss)
            losses.append(loss)
        # calculate target distribution for clustering in batch manner.
        # batchsize * k
        C_distribs = torch.stack(C_distribs)
        # batchsize * k
        C_distribs_sq = torch.pow(C_distribs, 2)
        # 1*k
        C_distribs_sum = torch.sum(C_distribs, dim=0, keepdim=True)
        # batchsize * k
        temp = C_distribs_sq / C_distribs_sum
        # batchsize * 1
        temp_sum = torch.sum(temp, dim=1, keepdim=True)
        target_distribs = temp / temp_sum
        # calculate the kl loss
        clustering_loss = self._lambda * F.kl_div(C_distribs.log(), target_distribs, reduction='batchmean')
        #print('The clustering loss is %.6f' % (clustering_loss.item()))
        np_losses_mean = torch.stack(losses).mean(0)
        total_loss = np_losses_mean + clustering_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), C_distribs.cpu().detach().numpy()

    def query_rec(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = 1
        # used for calculating the rmse.
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            #query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            query_set_y_pred = self.np(support_set_xs[i], support_set_ys[i], query_set_xs[i], query_set_ys[i])
            # obtain the mean of gaussian distribution
            #(interation_size, y_dim)
            #query_set_y_pred = query_set_y_pred.loc.detach()
            #print('test_y_pred size is ', query_set_y_pred.size())
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        output_list, recommendation_list = query_set_y_pred.view(-1).sort(descending=True)
        return losses_q.item(), recommendation_list


