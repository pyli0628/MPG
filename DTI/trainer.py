import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader, DataListLoader,Batch
from torch_geometric.nn import DataParallel
from sklearn import metrics
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import torch_geometric

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score


# binary class
class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        target=target.float()
        pt = torch.softmax(input, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()

#multi class
class FocalLoss_multi(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss_multi, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1 - prob,
                                                           self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class Trainer():
    def __init__(self, option, model,
                 train_dataset=None, valid_dataset=None, test_dataset=None,weight=[[1.0,1.0]],tasks_num=1):
        self.option = option
        # self.tasks = ["MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713",
        #               "MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"]
        self.tasks_num = tasks_num


        self.save_path = self.option['exp_path']

        self.device = torch.device("cuda:{}".format(0) \
                                       if torch.cuda.is_available() and not option['cpu']  else "cpu")
        self.model = DataParallel(model).to(self.device) \
            if option['parallel'] else model.to(self.device)

        #Setting the train valid and test data loader
        if train_dataset and valid_dataset:
            if self.option['parallel']:
                self.train_dataloader = DataListLoader(train_dataset, \
                                                       batch_size=self.option['batch_size'],shuffle=True)
                self.valid_dataloader = DataListLoader(valid_dataset, batch_size=self.option['batch_size'])
                if test_dataset: self.test_dataloader = DataListLoader(test_dataset, batch_size=self.option['batch_size'])
            else:
                self.train_dataloader = DataLoader(train_dataset, \
                                                   batch_size=self.option['batch_size'],shuffle=True,num_workers=4)
                self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.option['batch_size'],num_workers=4)
                if test_dataset: self.test_dataloader = DataLoader(test_dataset, batch_size=self.option['batch_size'],num_workers=4)
        else:
            self.test_dataset = test_dataset
            if self.option['parallel']:
                self.test_dataloader = DataListLoader(test_dataset, batch_size=self.option['batch_size'],num_workers=0)
            else:
                self.test_dataloader = DataLoader(test_dataset, batch_size=self.option['batch_size'],num_workers=4)

        # Setting the Adam optimizer with hyper-param

        if not option['focalloss']:
            self.criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(self.device),reduction='mean') for w in weight]
        else:
            self.log('Using FocalLoss')
            self.criterion = [FocalLoss(alpha=1/w[0]) for w in weight] #alpha 0.965
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option['lr'],
                                          weight_decay=option['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7,
            patience=self.option['lr_scheduler_patience'], min_lr=1e-6
        )
        self.start = time.time()
        self.records = {'best_epoch': None,'val_auc':[],
                        'best_val_auc':0.,'best_trn_auc':0.,'best_test_auc':0.}
        self.log(msgs=['\t{}:{}\n'.format(k, v) for k, v in self.option.items()],show=False)
        if train_dataset:
            self.log('train set num:{}    valid set num:{}    test set num: {}'.format(
                len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        self.log(msgs=str(model).split('\n'),show=False)

    def train_iterations(self):
        self.model.train()
        losses = []
        y_out_all = []
        y_pred_list = {}
        y_label_list = {}
        for data in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()

            if not self.option['parallel']:

                data = data.to(self.device)
                target_len = data.pr_len
                y_idx = torch.zeros(target_len[0]).long()
                for i, e in enumerate(target_len):
                    if i > 0:
                        y_idx = torch.cat([y_idx, torch.full((e.item(),), i).long()])
                y_idx = y_idx.to(self.device)
                data.protein = torch_geometric.utils.to_dense_batch(data.protein, y_idx)[0]

                output = self.model(data)
            else:
                output = self.model(data)
                data = Batch.from_data_list(data).to(self.device)
            loss=0
            for i in range(self.tasks_num):
                y_pred = output
                y_label = data.y

                loss+=self.criterion[i](y_pred, y_label)

                probs = F.softmax(y_pred.detach().cpu(), dim=-1)

                y_out = probs.argmax(dim=1).numpy()
                y_out_all.extend(y_out)
                y_pred = probs[:, 1].view(-1).numpy()


                # print(i,np.isnan(y_pred).any())
                try:
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                # print(np.isnan(y_label_list[i]))


            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        val_precision = metrics.precision_score(y_label_list[0],np.array(y_out_all))
        val_recall = metrics.recall_score(y_label_list[0], np.array(y_out_all))
        trn_roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(self.tasks_num)]
        trn_loss = np.array(losses).mean()

        return trn_loss,np.array(trn_roc).mean(),val_precision,val_recall

    def valid_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test' or mode=='eval': dataloader = self.test_dataloader
        if mode == 'valid': dataloader = self.valid_dataloader
        losses = []
        y_out_all = []
        y_pred_list = {}
        y_label_list = {}
        with torch.no_grad():
            for data in tqdm(dataloader):
                if not self.option['parallel']:
                    data = data.to(self.device)
                    target_len = data.pr_len
                    y_idx = torch.zeros(target_len[0]).long()
                    for i, e in enumerate(target_len):
                        if i > 0:
                            y_idx = torch.cat([y_idx, torch.full((e.item(),), i).long()])
                    y_idx = y_idx.to(self.device)
                    data.protein = torch_geometric.utils.to_dense_batch(data.protein, y_idx)[0]
                    output = self.model(data)
                else:
                    output = self.model(data)
                    data = Batch.from_data_list(data).to(self.device)
                loss = 0
                for i in range(self.tasks_num):
                    y_pred = output
                    y_label = data.y
                    loss = self.criterion[i](y_pred, y_label)
                    probs = F.softmax(y_pred.detach().cpu(), dim=-1)
                    y_out = probs.argmax(dim=1).numpy()
                    y_out_all.extend(y_out)
                    y_pred = probs[:, 1].view(-1).numpy()

                    try:
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    except:
                        y_label_list[i] = []
                        y_pred_list[i] = []
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    losses.append(loss.item())


        val_roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(self.tasks_num)]
        val_loss = np.array(losses).mean()

        val_precision = metrics.precision_score(y_label_list[0],np.array(y_out_all))
        val_recall = metrics.recall_score(y_label_list[0], np.array(y_out_all))


        return val_loss,np.array(val_roc).mean(),val_precision,val_recall



    def train(self):
        self.log('Training start...')
        early_stop_cnt = 0
        for epoch in range(self.option['epochs']):
            trn_loss,trn_roc, trn_pre,trn_recall = self.train_iterations()
            val_loss,val_roc, val_pre,val_recall = self.valid_iterations()
            test_loss,test_roc,test_pre,test_recall = self.valid_iterations(mode='test')

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']
            self.log('Epoch:{} {} trn_loss:{:.3f} trn_roc:{:.3f} trn_precision:{:.3f} trn_recall:{:.3f} lr_cur:{:.5f}'.
                     format(epoch,self.option['dataset'], trn_loss,trn_roc,trn_pre,trn_recall, lr_cur),
                     with_time=True)
            self.log('Epoch:{} {} val_loss:{:.3f} val_roc:{:.3f} val_precision:{:.3f} val_recall:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], val_loss, val_roc, val_pre, val_recall, lr_cur),
                     with_time=True)
            self.log('Epoch:{} {} test_loss:{:.3f} test_roc:{:.3f} test_precision:{:.3f} test_recall:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], test_loss, test_roc, test_pre, test_recall, lr_cur),
                     with_time=True)
            self.records['val_auc'].append(val_roc)
            if val_roc == np.array(self.records['val_auc']).max() :
                self.save_model_and_records(epoch,test_pre,test_recall, test_roc, final_save=False)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if self.option['early_stop_patience'] > 0 and early_stop_cnt > self.option['early_stop_patience']:
                self.log('Early stop hitted!')
                break
        self.log('The best epoch is {}, test_presion: {}, test_recall: {}, test_auc: {}'.format(self.records['best_epoch'],
                                                                                       self.records['best_test_precision'],
                                                                                       self.records['best_test_recall'],
                                                                                       self.records['best_test_auc']))
        return self.records['best_test_auc']


    def predict(self):
        self.model.eval()
        dataloader = self.test_dataloader
        ret = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                if not self.option['parallel']:
                    data = data.to(self.device)
                    target_len = data.pr_len
                    y_idx = torch.zeros(target_len[0]).long()
                    for i, e in enumerate(target_len):
                        if i > 0:
                            y_idx = torch.cat([y_idx, torch.full((e.item(),), i).long()])
                    y_idx = y_idx.to(self.device)
                    data.protein = torch_geometric.utils.to_dense_batch(data.protein, y_idx)[0]
                    output = self.model(data)
                else:
                    output = self.model(data)

                output = F.softmax(output.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                ret.extend(output)


        return ret



    def save_model_and_records(self, epoch, test_pre,test_recall,test_auc=None, final_save=False):
        if final_save:
            # self.save_loss_records()
            file_name = 'best_model_{}.ckpt'.format(self.option['seed'])
        else:
            file_name = 'best_model_{}.ckpt'.format(self.option['seed'])
            self.records['best_epoch'] = epoch
            self.records['best_test_precision'] =  test_pre
            self.records['best_test_recall'] =test_recall
            self.records['best_test_auc'] = test_auc

        if not self.option['parallel']:
           model_dic = self.model.state_dict()
        else:
           model_dic = self.model.module.state_dict()

        with open(os.path.join(self.save_path, file_name), 'wb') as f:
            torch.save(model_dic, f)
        self.log('Model saved at epoch {}'.format(epoch))


    def save_loss_records(self):
        trn_record = pd.DataFrame(self.records['trn_record'],
                                  columns=['epoch', 'trn_loss','trn_auc','trn_acc', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'],
                                  columns=['epoch', 'val_loss','val_auc','val_acc', 'lr'])
        ret = pd.DataFrame({
            'epoch': trn_record['epoch'],
            'trn_loss': trn_record['trn_loss'],
            'val_loss': val_record['val_loss'],
            'trn_auc':trn_record['trn_auc'],
            'val_auc':val_record['val_auc'],
            'trn_lr':trn_record['lr'],
            'val_lr':val_record['lr']
        })
        ret.to_csv(self.save_path+'/record.csv')
        return ret

    def load_best_ckpt(self):
        ckpt_path = self.save_path + '/' + self.records['best_ckpt']
        self.log('The best ckpt is {}'.format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log('Ckpt loading: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.option = ckpt['option']
        self.records = ckpt['records']
        self.model.load_state_dict(ckpt['model_state_dict'])

    def log(self, msg=None, msgs=None, with_time=False,show=True):
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.save_path+'/log.txt', 'a+') as f:
            if msgs:
                self.log('#' * 80)
                if '\n' not in msgs[0]: msgs = [m + '\n' for m in msgs]
                f.writelines(msgs)
                if show:
                    for x in msgs:
                        print(x, end='')
                self.log('#' * 80)
            if msg:
                f.write(msg + '\n')
                if show:
                    print(msg)



class Trainer_old():
    def __init__(self, option, model,
                 train_dataset=None, valid_dataset=None, test_dataset=None,weight=[[1.0,1.0]],tasks_num=1):
        self.option = option
        # self.tasks = ["MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713",
        #               "MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"]
        self.tasks_num = tasks_num


        self.save_path = self.option['exp_path']

        self.device = torch.device("cuda:{}".format(0) \
                                       if torch.cuda.is_available() else "cpu")
        self.model = DataParallel(model).to(self.device) \
            if option['parallel'] else model.to(self.device)

        #Setting the train valid and test data loader
        if train_dataset and valid_dataset:
            if self.option['parallel']:
                self.train_dataloader = DataListLoader(train_dataset, \
                                                       batch_size=self.option['batch_size'],shuffle=True)
                self.valid_dataloader = DataListLoader(valid_dataset, batch_size=self.option['batch_size'])
                if test_dataset: self.test_dataloader = DataListLoader(test_dataset, batch_size=self.option['batch_size'])
            else:
                self.train_dataloader = DataLoader(train_dataset, \
                                                   batch_size=self.option['batch_size'],shuffle=True)
                self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.option['batch_size'])
                if test_dataset: self.test_dataloader = DataLoader(test_dataset, batch_size=self.option['batch_size'])
        else:
            self.test_dataset = test_dataset
            if self.option['parallel']:
                self.test_dataloader = DataListLoader(test_dataset, batch_size=self.option['batch_size'],num_workers=0)
            else:
                self.test_dataloader = DataLoader(test_dataset, batch_size=self.option['batch_size'],num_workers=0)

        # Setting the Adam optimizer with hyper-param

        if not option['focalloss']:
            # self.criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(self.device),reduction='mean') for w in weight]
            self.criterion = [torch.nn.CrossEntropyLoss(reduction='mean')]
        else:
            self.log('Using FocalLoss')
            self.criterion = [FocalLoss(alpha=1/w[0]) for w in weight] #alpha 0.965
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option['lr'],
                                          weight_decay=option['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7,
            patience=self.option['lr_scheduler_patience'], min_lr=1e-6
        )
        self.start = time.time()
        self.records = {'best_epoch': None,'val_auc':[],
                        'best_val_auc':0.,'best_trn_auc':0.,'best_test_auc':0.}
        self.log(msgs=['\t{}:{}\n'.format(k, v) for k, v in self.option.items()],show=False)
        if train_dataset:
            self.log('train set num:{}    valid set num:{}    test set num: {}'.format(
                len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        self.log(msgs=str(model).split('\n'),show=False)

    def train_iterations(self):
        self.model.train()
        losses = []
        y_out_all = []
        y_pred_list = {}
        y_label_list = {}
        for data in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()

            if not self.option['parallel']:
                data = data.to(self.device)
                output = self.model(data)
            else:
                output = self.model(data)
                data = Batch.from_data_list(data).to(self.device)
            loss=0
            for i in range(self.tasks_num):
                y_pred = output
                y_label = data.y

                loss+=self.criterion[i](y_pred, y_label)

                probs = F.softmax(y_pred.detach().cpu(), dim=-1)

                y_out = probs.argmax(dim=1).numpy()
                y_out_all.extend(y_out)
                y_pred = probs[:, 1].view(-1).numpy()

                # print(i,np.isnan(y_pred).any())
                try:
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                # print(np.isnan(y_label_list[i]))


            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        val_precision = metrics.precision_score(y_label_list[0],np.array(y_out_all))
        val_recall = metrics.recall_score(y_label_list[0], np.array(y_out_all))
        trn_roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(self.tasks_num)]
        trn_loss = np.array(losses).mean()



        return trn_loss,np.array(trn_roc).mean(),val_precision,val_recall

    def valid_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test' or mode=='eval': dataloader = self.test_dataloader
        if mode == 'valid': dataloader = self.valid_dataloader
        losses = []
        y_out_all = []
        y_pred_list = {}
        y_label_list = {}
        with torch.no_grad():
            for data in tqdm(dataloader):
                if not self.option['parallel']:
                    data = data.to(self.device)
                    output = self.model(data)
                else:
                    output = self.model(data)
                    data = Batch.from_data_list(data).to(self.device)
                loss = 0
                for i in range(self.tasks_num):
                    y_pred = output
                    y_label = data.y

                    loss = self.criterion[i](y_pred, y_label)

                    probs = F.softmax(y_pred.detach().cpu(), dim=-1)

                    y_out = probs.argmax(dim=1).numpy()
                    y_out_all.extend(y_out)
                    y_pred = probs[:, 1].view(-1).numpy()

                    try:
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    except:
                        y_label_list[i] = []
                        y_pred_list[i] = []
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    losses.append(loss.item())

        val_roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(self.tasks_num)]
        val_loss = np.array(losses).mean()

        val_precision = metrics.precision_score(y_label_list[0],np.array(y_out_all))
        val_recall = metrics.recall_score(y_label_list[0], np.array(y_out_all))


        # if mode=='eval':
        #     self.log('SEED {} DATASET {}  The best test_loss:{:.3f} test_roc:{:.3f} precision:{:.3f} recall:{:.3f}.'
        #              .format(self.option['seed'],self.option['dataset'],val_loss, np.array(val_roc).mean(),val_precision,val_recall))
        #
        #

        return val_loss,np.array(val_roc).mean(),val_precision,val_recall



    def train(self):
        self.log('Training start...')
        early_stop_cnt = 0
        for epoch in range(self.option['epochs']):
            trn_loss, trn_roc, trn_pre, trn_recall = self.train_iterations()
            val_loss, val_roc, val_pre, val_recall = self.valid_iterations()
            test_loss, test_roc, test_pre, test_recall = self.valid_iterations(mode='test')

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']
            self.log('Epoch:{} {} trn_loss:{:.3f} trn_roc:{:.3f} trn_precision:{:.3f} trn_recall:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], trn_loss, trn_roc, trn_pre, trn_recall, lr_cur),
                     with_time=True)
            self.log('Epoch:{} {} val_loss:{:.3f} val_roc:{:.3f} val_precision:{:.3f} val_recall:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], val_loss, val_roc, val_pre, val_recall, lr_cur),
                     with_time=True)
            self.log(
                'Epoch:{} {} test_loss:{:.3f} test_roc:{:.3f} test_precision:{:.3f} test_recall:{:.3f} lr_cur:{:.5f}'.
                format(epoch, self.option['dataset'], test_loss, test_roc, test_pre, test_recall, lr_cur),
                with_time=True)
            self.records['val_auc'].append(val_roc)
            if val_roc == np.array(self.records['val_auc']).max():
                self.save_model_and_records(epoch, test_pre, test_recall, test_roc, final_save=False)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if self.option['early_stop_patience'] > 0 and early_stop_cnt > self.option['early_stop_patience']:
                self.log('Early stop hitted!')
                break
        self.log(
            'The best epoch is {}, test_presion: {}, test_recall: {}, test_auc: {}'.format(self.records['best_epoch'],
                                                                                           self.records[
                                                                                               'best_test_precision'],
                                                                                           self.records[
                                                                                               'best_test_recall'],
                                                                                           self.records[
                                                                                               'best_test_auc']))
        return self.records['best_test_auc']


    def predict(self):
        self.model.eval()
        dataloader = self.test_dataloader
        ret = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                if not self.option['parallel']:
                    data = data.to(self.device)
                    output = self.model(data)
                else:
                    output = self.model(data)

                output = F.softmax(output.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                ret.extend(output)
        smiles = self.test_dataset.smiles
        out_df = pd.DataFrame({'smiles':smiles,'prediction':ret})
        out_df.to_csv(os.path.join(self.save_path,'{}_prediction.csv'.format(self.test_dataset.dataset)))

        return ret



    def save_model_and_records(self, epoch, test_pre,test_recall,test_auc=None, final_save=False):
        if final_save:
            # self.save_loss_records()
            file_name = 'best_model_{}.ckpt'.format(self.option['seed'])
        else:
            file_name = 'best_model_{}.ckpt'.format(self.option['seed'])
            self.records['best_epoch'] = epoch
            self.records['best_test_precision'] =  test_pre
            self.records['best_test_recall'] =test_recall
            self.records['best_test_auc'] = test_auc

        if not self.option['parallel']:
           model_dic = self.model.state_dict()
        else:
           model_dic = self.model.module.state_dict()

        with open(os.path.join(self.save_path, file_name), 'wb') as f:
            torch.save({
                'option': self.option,
                'records': self.records,
                'model_state_dict': model_dic,
            }, f)
        self.log('Model saved at epoch {}'.format(epoch))


    def save_loss_records(self):
        trn_record = pd.DataFrame(self.records['trn_record'],
                                  columns=['epoch', 'trn_loss','trn_auc','trn_acc', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'],
                                  columns=['epoch', 'val_loss','val_auc','val_acc', 'lr'])
        ret = pd.DataFrame({
            'epoch': trn_record['epoch'],
            'trn_loss': trn_record['trn_loss'],
            'val_loss': val_record['val_loss'],
            'trn_auc':trn_record['trn_auc'],
            'val_auc':val_record['val_auc'],
            'trn_lr':trn_record['lr'],
            'val_lr':val_record['lr']
        })
        ret.to_csv(self.save_path+'/record.csv')
        return ret

    def load_best_ckpt(self):
        ckpt_path = self.save_path + '/' + self.records['best_ckpt']
        self.log('The best ckpt is {}'.format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log('Ckpt loading: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.option = ckpt['option']
        self.records = ckpt['records']
        self.model.load_state_dict(ckpt['model_state_dict'])

    def log(self, msg=None, msgs=None, with_time=False,show=True):
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.save_path+'/log.txt', 'a+') as f:
            if msgs:
                self.log('#' * 80)
                if '\n' not in msgs[0]: msgs = [m + '\n' for m in msgs]
                f.writelines(msgs)
                if show:
                    for x in msgs:
                        print(x, end='')
                self.log('#' * 80)
            if msg:
                f.write(msg + '\n')
                if show:
                    print(msg)



