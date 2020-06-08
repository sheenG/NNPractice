# coding=utf-8
from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides
from utils import load_jsonl
import time
import sys
import pickle
import os
import pandas

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

    def show(self):
        print(self.learning_rate)


model_config = Config(
    testing=True,
    seed=1,
    batch_size=4,
    learning_rate=0.00002,
    #learning_rate=0.0001,
    epochs=10,
    patience=2,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
    max_candi_num = 10
)

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.tokenizers import Token

from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField

class ListWiseDatasetReader(DatasetReader):
    def __init__(self, input_mode = 0, max_seq_len=model_config.max_seq_len,
                 max_candi_num=model_config.max_candi_num):
        super().__init__(lazy=False)
        self.token_indexer = PretrainedBertIndexer(
            pretrained_model = "bert-base-uncased",
            max_pieces = model_config.max_seq_len,
            do_lowercase = True
        )
        self.input_mode = input_mode
        print(type(self.input_mode))
        self.token_indexers = {"tokens": self.token_indexer}
        self.max_seq_len = max_seq_len
        self.max_candi_num = max_candi_num

    def text_to_instances(self, input_token_list,
                in_out_mat,
                concat_pairs_token_list,
                ranking_feature_list,
                head_mat,
                tail_mat,
                clues_sum,
                candi_sum,
                id,
                label):
        # TODO:改写这个函数
        # all_concerned_input_tokens
        input_token_field = ListField([TextField(tokens, self.token_indexers) for tokens in input_token_list])
        fields = {"input_token_field": input_token_field}

        in_out_field = ArrayField(array=in_out_mat)
        fields["in_out_field"] = in_out_field

        concat_pairs_field = ListField([TextField(tokens, self.token_indexers) for tokens in concat_pairs_token_list])
        fields["concat_pairs_field"] = concat_pairs_field

        ranking_feature_field = ArrayField(array=np.array(ranking_feature_list))
        fields["ranking_feature_field"] = ranking_feature_field

        head_field = ArrayField(array=head_mat)
        fields["head_field"] = head_field

        tail_field = ArrayField(array=tail_mat)
        fields["tail_field"] = tail_field

        '''
        print(in_out_mat)
        print(ranking_feature_list)
        print(head_mat)
        print(tail_mat)
        exit(0)
        '''

        clues_sum_field = ArrayField(array=np.array([clues_sum]))
        fields["clues_sum_field"] = clues_sum_field

        candi_sum_field = ArrayField(array=np.array([candi_sum]))
        fields["candi_sum_field"] = candi_sum_field
        #print(sent_sum)

        id_field = MetadataField(id)
        fields["id"] = id_field

        label_field = ArrayField(array=np.array([label]))
        fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path) -> Iterator[Instance]:
        # TODO:读取数据
        data = load_jsonl(file_path)
        suffix = file_path.split("_")[-1].split('.')[0]
        #print(file_path)
        #print(suffix)
        gcn_path = "../data/gcn_fold_dir/gcn_{}.pkl".format(suffix)
        gcn_mat_list = pickle.load(open(gcn_path, "rb"))
        for data_idx, item in enumerate(data):
            # 将claim fact、clue fact、true candi、false candi都读进来
            sent_list = []

            claim_facts = item["claim_facts"]
            sent_list.append(claim_facts[0])

            clues_facts = item["clues_facts"]
            clues_whole_seq = ""

            # TODO：clues 和 candi都加上claim，但是，使用单句建模，不用句对建模
            # 单句建模结果侧重描述语义，而句对建模结果更强调两个子部分的判别关系
            #print(self.input_mode)
            for clue in clues_facts:
                if self.input_mode == 0:
                    sent_list.append(claim_facts[0] + " [SEP] " + clue[0])
                elif self.input_mode == 1:
                    sent_list.append(claim_facts[0] + clue[0])
                clues_whole_seq += clue[0]

            candi_facts = item["candi_facts"]
            for candi in candi_facts:
                if self.input_mode == 0:
                    sent_list.append(claim_facts[0] + " [SEP] " + candi[0])
                elif self.input_mode == 1:
                    sent_list.append(claim_facts[0] + candi[0])

            label = item["label"]
            id = item["id"]

            # self.tokenizer是PretrainedBertIndexer，会自动首尾补充[CLS]和[SEP]
            input_token_list = [[Token(x) for x in self.tokenizer(str(sent))]
                                    for sent in sent_list]

            concat_sent_pairs_list = []
            for candi in candi_facts:
                concat_sent_pairs_list.append(claim_facts[0] + " [SEP] " + candi[0] + " " + clues_whole_seq)

            concat_pairs_token_list = [[Token(x) for x in self.tokenizer(str(sent))]
                                       for sent in concat_sent_pairs_list]

            gcn_info = gcn_mat_list[data_idx]
            in_out_mat = gcn_info["in_out"]
            ranking_feature_list = gcn_info["ranking_feature"]
            head_mat = gcn_info["head"]
            tail_mat = gcn_info["tail"]

            clues_sum = len(clues_facts)
            candi_sum = len(candi_facts)

            yield self.text_to_instances(
                input_token_list,
                in_out_mat,
                concat_pairs_token_list,
                ranking_feature_list,
                head_mat,
                tail_mat,
                clues_sum,
                candi_sum,
                id,
                label
            )

    def tokenizer(self, sent: str):
        # 这里使用word_piece形式切分，得到的token 粒度更小
        return self.token_indexer.wordpiece_tokenizer(sent)

from allennlp.models import Model
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


bert_embedder = PretrainedBertEmbedder(
            pretrained_model="bert-base-uncased",
            top_layer_only=True,
            requires_grad=True
)

word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                 allow_unmatched_keys = True)

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
class bertSentencePooler(Seq2VecEncoder):
    def forward(self, embs: torch.tensor, mask: torch.tensor = None):
        # 返回[CLS]对应的隐向量
        # 这里的输入shape是[batch_size, seq_num, seq_len, bert_768]
        # 因此输出shape是[batch_size, seq_num, cls_768]
        return embs[:, :, 0]
    def get_output_dim(self):
        return word_embeddings.get_output_dim()

from torch import nn
from allennlp.data.vocabulary import Vocabulary
from collections import OrderedDict
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_log_softmax, masked_softmax, masked_mean
from torch.nn.functional import nll_loss, hinge_embedding_loss, pad

class BERTModel(Model):
    # 这里默认使用bert_word_embedder
    # 此模型基于bert，由于是直接获取CLS向量作为句子向量表示，因此和ELMO等注重词建模的模型很不同
    # 所以此模型中的interaction与CDR_elmo模型代码不复用
    def __init__(self, word_embed: TextFieldEmbedder = word_embeddings,
                    bert_sent_len = 768):
        # 针对每个输入text passage，做embedding，或者输入CDR之前就已经用bert转换成embedding vectors
        super(BERTModel, self).__init__(Vocabulary())
        vocab = Vocabulary()
        self.use_hand_fea = True
        self.use_global = True
        self.embedder = word_embed
        self.encoder = bertSentencePooler(vocab)
        #self.interaction = Interaction(claim_shape, clues_shape, candi_shape)
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(bert_sent_len, 100)),
            ('activation1', nn.ReLU(inplace=True)),
            #('softmax', nn.Softmax(dim=-1))
            #('tanh2', nn.Tanh())
        ]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(bert_sent_len, 100)),
            ('activation2', nn.ReLU(inplace=True)),
            #('tanh2', nn.Tanh())
        ]))
        self.proj_1 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(bert_sent_len, 100 * 2)),
            ('activation3', nn.ReLU(inplace=True))
            #('activation3', nn.Tanh())
        ]))
        if self.use_hand_fea:
            self.proj_2 = nn.Sequential(OrderedDict([
                ('fc4', nn.Linear(200 * 4, 100)),
                ('activation4', nn.ReLU(inplace=True)),
                #('tanh5', nn.Tanh())
            ]))
            self.prediction = nn.Sequential(OrderedDict([
                ('fc5', nn.Linear(100 + 8, 1)),
            ]))
        else:
            self.prediction = nn.Sequential(OrderedDict([
                ('fc6', nn.Linear(100, 1)),
                #('fc4', nn.Linear(300 * 4, 100)),
            ]))
        if self.use_global:
            self.bert_prediction = nn.Sequential(OrderedDict([
                ('fc7', nn.Linear(bert_sent_len + 100 + 8, 100)),
                ('activation', nn.ReLU(inplace=True)),
                ('fc8', nn.Linear(100, 1))
                #('softmax', nn.Softmax(dim=-1))
            ]))
        '''
        self.bert_prediction = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(bert_sent_len + 100 + 8, 1)),
            ('activation', nn.ReLU(inplace=True)),
        ]))
        '''

        self.accuracy = CategoricalAccuracy()
        self.loss_type = "m-hinge-loss"
        #self.loss_type = "CE"
        pass

    def forward(self, input_token_field, in_out_field,
                concat_pairs_field,
                ranking_feature_field,
                    head_field, tail_field, clues_sum_field, candi_sum_field, id, label):
        # 这里函数参数名应当与读数据实例化时字典的key匹配
        # 因此需要传递多组不同参数时，使用不同field中不同key即可
        # bert CLS encoder stage
        # claim_vectors_shape : [batch_size, claim_fact_num, bert_vector_len]

        # concat_pairs_field为一个字典，key包含"tokens"、"mask"等
        # get_text_field_mask这里是直接返回原本的mask
        '''
        print(ranking_feature_field)
        print(ranking_feature_field.shape)
        exit(0)
        '''

        input_mask = get_text_field_mask(input_token_field)
        input_embeddings = self.embedder(input_token_field, num_wrapping_dims=1)
        input_vectors = self.encoder(input_embeddings, input_mask)
        device_id = input_vectors.get_device()
        #print(device_id)
        '''
        print()
        print(input_mask.shape)
        print(input_embeddings.shape)
        print(input_vectors.shape)
        print(clues_sum_field)
        print(candi_sum_field)
        print(in_out_field)
        '''

        # 计算拉普拉斯矩阵
        # 当图结构中为无向边，使用对称归一化拉普拉斯矩阵
        # 当图结构中为有向边时，使用非对称归一化的拉普拉斯矩阵

        # 使用in_out矩阵构造非对称归一化拉普拉斯矩阵
        # 信息融合不应该泄露？只考虑clues和candi
        # 因此在这里首先将claim位置全部抹掉
        claim_tensor = input_vectors[:, 0, :]
        evidence_part_tensor = input_vectors[:, 1:, :]

        # initialization
        # 截取
        degree_mat_1 = in_out_field[:, 1:, 1:].clone()
        # 将所有对角元素置为1
        #print(degree_mat_1.shape)
        #print(degree_mat_1)
        identity = torch.eye(degree_mat_1.shape[-1]).cuda(device_id)
        #print(identity)
        identity_batch = identity.reshape((1, identity.shape[0], identity.shape[1])).repeat(
            degree_mat_1.shape[0], 1, 1
        )
        #print(identity_batch.shape)
        #print(identity_batch)
        # TODO:加上条件
        degree_mat_1[(degree_mat_1 > 0) | (identity_batch > 0)] = 1
        #print(degree_mat_1)

        degree_mat_1.requires_grad = False
        # 由于对角元素始终为1，因此sum必不为0
        row_sum = degree_mat_1.sum(dim = -1, keepdim = True).repeat(1, 1, degree_mat_1.shape[-1])
        col_sum = degree_mat_1.sum(dim = -2, keepdim = True).repeat(1, degree_mat_1.shape[-2], 1)
        in_laplacian_1 = torch.div(degree_mat_1, row_sum)
        out_laplacian_1 = torch.div(degree_mat_1, col_sum)

        #print(row_sum)
        #print(col_sum)
        #print(in_laplacian_1.shape)
        #print(in_laplacian_1[-1])
        #print(out_laplacian_1.shape)
        #print(out_laplacian_1[-1])

        # 这个矩阵代表结构信息，不更新
        in_laplacian_1.requires_grad = False
        out_laplacian_1.requires_grad = False

        # 所有证据（clue和候选）共享lap矩阵时，若传播超过一层，由于信息流动，candi之间会有信息交互,
        # 先尝试一层传播
        in_H_1 = self.fc1(torch.bmm(in_laplacian_1, evidence_part_tensor))
        out_H_1 = self.fc2(torch.bmm(out_laplacian_1, evidence_part_tensor))

        # 拼接再映射，也可以考虑加和，看情况
        concat_hidden = torch.cat((in_H_1, out_H_1), -1)
        #print(concat_hidden.shape)


        # 根据clues_sum_field,candi_sum_field
        padding_vec = torch.Tensor([0] * concat_hidden.shape[-1]).unsqueeze(0).cuda(device_id)
        # 从concat_hidden中丢弃clues
        # 因此在做padding时，最大数量为原始concat_hidden的句子数量维度 - max(clues_sum)
        # 例如：在concat_hidden中，
        # group1中，clues ： candi ： padding = 3 ： 2 ： 1
        # group2中，clues ： candi ： padding = 4 ： 2 ： 0
        # group3中，clues ： candi ： padding = 1 ： 5 ： 0
        # 表示candi矩阵最多应当容纳5条candidate，多余的位置padding
        # 则candi_max_size = 5
        # 以这个值做重新填充时，padd1 = 5 - (6 - 3) = 2
        # padd2 = 5 - (6 - 4) = 3
        # padd3 = 5 - (6 - 1) = 0
        # TODO:做具体值的check，验证得到的candi_hidden是否和concat_hidden中一致
        candi_max_size = torch.max(candi_sum_field).long().item()
        #print(candi_max_size)

        # 是否需要重新padding if 需要 else 不需要
        candi_hidden = torch.stack([torch.cat((batch[clues_sum_field[idx].long().item():
                                    clues_sum_field[idx].long().item() + candi_sum_field[idx].long().item()],
                            padding_vec.repeat(candi_max_size - candi_sum_field[idx].long().item(), 1)), dim=0)
                        if candi_max_size > candi_sum_field[idx].long().item()
                        else batch[clues_sum_field[idx].long().item():
                                    clues_sum_field[idx].long().item() + candi_sum_field[idx].long().item()]
                        for idx, batch in enumerate(concat_hidden)], dim = 0)
        '''
        print(candi_hidden)
        print(candi_hidden.shape)
        print(clues_sum_field)
        print(candi_sum_field)
        exit(0)
        '''
        # 与claim做alignment
        projected_claim = self.proj_1(claim_tensor).unsqueeze(1).repeat(1, candi_hidden.shape[1], 1)
        #print(candi_hidden.shape)
        #print(projected_claim.shape)
        if self.use_hand_fea:
            #aligned = torch.cat([candi_hidden, projected_claim,
            #                     candi_hidden - projected_claim, candi_hidden * projected_claim,
            #    pad(ranking_feature_field, (0, 0, 0, candi_hidden.shape[1] - ranking_feature_field.shape[1]))], -1)
            aligned = torch.cat([candi_hidden, projected_claim,
                                 candi_hidden - projected_claim, candi_hidden * projected_claim,
                ], -1)
        else:
            aligned = torch.cat([candi_hidden, projected_claim,
                                 candi_hidden - projected_claim, candi_hidden * projected_claim], -1)
        #fc_out = self.proj_2(aligned).squeeze(-1)
        fc_out1 = self.proj_2(aligned)

        if not self.use_global:
            fc_out = self.prediction(torch.cat([fc_out1,
                pad(ranking_feature_field, (0, 0, 0, candi_hidden.shape[1] - ranking_feature_field.shape[1]))], -1))
            fc_out = fc_out.squeeze(-1)

            # 根据clues与candi个数生成softmax_mask
            # 这里mask的padding方式和candi_hidden padding稍有不同，但是目的是一致的
            softmax_mask = torch.Tensor([[1] * int(candi_sum_field[idx].long().item()) +
                                         [0] * int(candi_max_size - candi_sum_field[idx].long().item())
                                         for idx in range(candi_sum_field.shape[0])]).cuda(device_id)
            #print(softmax_mask)
            #print(candi_sum_field)
            #exit(0)
        ############################### global interaction ########################################
        else:
            pairs_mask = get_text_field_mask(concat_pairs_field)
            pairs_embeddings = self.embedder(concat_pairs_field, num_wrapping_dims=1)
            pairs_vectors = self.encoder(pairs_embeddings, pairs_mask)
            '''
            print()
            print(pairs_mask.shape)
            print(pairs_embeddings.shape)
            print(pairs_vectors.shape)
            '''

            #fc_out = self.bert_prediction(pairs_vectors).squeeze(-1)
            '''
            print('candi sum')
            print(candi_sum_field)
            print(pairs_vectors.shape)
            print(fc_out1.shape)
            print(pad(ranking_feature_field,
                    (0, 0, 0, candi_hidden.shape[1] - ranking_feature_field.shape[1])).shape)
            print(softmax_mask)
            '''

            softmax_mask = pairs_mask.sum(-1)
            softmax_mask = (softmax_mask > 0)
            #print(softmax_mask)
            fc_out = self.bert_prediction(torch.cat([pairs_vectors, fc_out1,
                pad(ranking_feature_field,
                    (0, 0, 0, candi_hidden.shape[1] - ranking_feature_field.shape[1]))],-1)).squeeze(-1)

        ###########################################################################################

        logits = masked_log_softmax(fc_out, softmax_mask)
        output = {'logits': logits}
        #print(predictions.shape)
        label = label.squeeze(-1)

        if label is not None:
            mask = torch.ones(label.size(), device=device_id)
            self.accuracy(logits, label, mask)
            #TODO: multi negative sample hinge loss
            if self.loss_type == "m-hinge-loss":
                # 选出每个数据组中ground truth的得分，计算分数差，优化分数差
                # 参考链接：http://cs231n.github.io/linear-classify/#interpret
                softmax_logits = masked_softmax(fc_out, softmax_mask)
                #print(softmax_logits)
                #print(softmax_mask)
                #sorted_logits, indices = torch.sort(softmax_logits, dim=-1, descending=True)

                # 表示将ground truth 标签对应的分数选出来，用来计算margin
                gd_candi_score = softmax_logits.gather(-1, label.long().view(-1, 1))

                padding_pos_boolean = (softmax_mask == 0)
                # 在hinge loss 中，使用最大得分减去其他
                score_diff = move_to_device(gd_candi_score.view(gd_candi_score.shape[0], 1).repeat(1,
                                            softmax_logits.shape[-1]), device_id) - softmax_logits

                # 考虑pytorch hinge loss的计算过程，将padding位置的score_diff设置为0
                score_diff[padding_pos_boolean] = 0
                #print(score_diff)
                #print(score_diff.shape)
                hinge_label_tensor = move_to_device(torch.Tensor([-1]).repeat(softmax_logits.size()), device_id)
                #print(label_tensor.shape)

                pos_tensor = move_to_device(torch.Tensor(list(range(0, softmax_logits.shape[-1])) *
                                softmax_logits.shape[0]).view(softmax_logits.size()), device_id).long()
                #print(pos_tensor)
                gd_pos_boolean = (pos_tensor ==
                        (label.long().view(-1, 1).repeat(1, pos_tensor.shape[-1])))
                #print(label.long().view(-1, 1).repeat(1, pos_tensor.shape[-1]))
                #print(gd_pos_boolean)

                # 根据pytorch hinge loss函数，我们将不需要梯度回传的部分，标签设为1，对应分数设为0
                # 这样就对loss无贡献
                # 参考链接：https://pytorch.org/docs/stable/nn.html#torch.nn.HingeEmbeddingLoss
                hinge_label_tensor[gd_pos_boolean] = 1
                hinge_label_tensor[padding_pos_boolean] = 1

                #print(hinge_label_tensor)

                # 这里我们只关心应当得到结果的部分
                # mask遮住的地方，应当不考虑
                output['loss'] = hinge_embedding_loss(score_diff.view(-1), hinge_label_tensor.view(-1),
                                                      reduction='sum'
                                                      )
                #print(output['loss'])
                #input()
            else:
                output['loss'] = nll_loss(logits, label.long())

        return output

    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}


from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.nn.util import move_to_device

def bert_rank_train(k_fold = 0, input_mode=0):
    claim_list_len = 1
    # 注意，这里设置的clues_list_len，实际上传入模型后并没有用上，仍然是模型自己做padding，而不是截断为5个info句子
    clues_list_len = 5
    candi_list_len = 15
    bert_vector_len = 768

    data_fold = k_fold
    print("process data fold {}".format(data_fold))
    model = BERTModel(word_embeddings, bert_sent_len=bert_vector_len)

    if torch.cuda.is_available():
        cuda_device = list(range(torch.cuda.device_count()))

        model = model.cuda(cuda_device[0])
    else:
        cuda_device = -1
    print("cuda device : {}".format(cuda_device))

    reader = ListWiseDatasetReader(input_mode=input_mode)
    train_dataset = reader.read("../data/CDR/CDR_fold_dir/CDR_train{}.jsonl".format(data_fold))
    dev_dataset = reader.read("../data/CDR/CDR_fold_dir/CDR_dev{}.jsonl".format(data_fold))
    test_dataset = reader.read("../data/CDR/CDR_fold_dir/CDR_test{}.jsonl".format(data_fold))

    '''
    for p in model.parameters():
        if p.requires_grad:
            print(p.name)
    exit(0)
    '''

    fc_lr = 1e-3
    #optimizer = torch.optim.SGD(model.parameters(), lr=model_config.learning_rate, momentum=0.9)
    optimizer = torch.optim.SGD([{'params': model.embedder.parameters()},
                                 {'params': model.fc1.parameters(), 'lr': fc_lr},
                                 {'params': model.fc2.parameters(), 'lr': fc_lr},
                                 {'params': model.proj_1.parameters(), 'lr': fc_lr},
                                 {'params': model.proj_2.parameters(), 'lr': fc_lr},
                                 {'params': model.bert_prediction.parameters(), 'lr': fc_lr},
                                 ], lr=model_config.learning_rate, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    vocab = Vocabulary()
    iterator_train = BucketIterator(batch_size=model_config.batch_size, sorting_keys=[("input_token_field", "num_fields")])
    iterator_train.index_with(vocab)

    model.train()
    trainer = Trainer(model = model,
                      optimizer = optimizer,
                      iterator = iterator_train,
                      train_dataset = train_dataset,
                      validation_dataset = dev_dataset,
                      patience = model_config.patience,
                      num_epochs = model_config.epochs,
                      cuda_device = cuda_device,
                      shuffle=True
                      )
    trainer.train()

    # test
    model.eval()

    preds = []
    gd = []
    ids = []
    candi_sum = []
    gd_pos = []
    with torch.no_grad():
        iterator_test = BucketIterator(batch_size = model_config.batch_size, sorting_keys=[("input_token_field", "num_fields")])
        iterator_test.index_with(vocab)
        generator_test = iterator_test(test_dataset, 1, False)
        for batch in generator_test:
            batch = move_to_device(batch, cuda_device[0])
            gd.extend(batch['label'].squeeze(-1).long().cpu().numpy().tolist())
            out_dict = model(batch['input_token_field'], batch['in_out_field'],
                             batch['concat_pairs_field'], batch['ranking_feature_field'],
                             batch['head_field'], batch['tail_field'], batch['clues_sum_field'],
                             batch['candi_sum_field'], batch['id'], batch['label'])
            batch_pred = torch.argmax(out_dict['logits'], -1).cpu().numpy()
            preds.extend(batch_pred.tolist())
            ids.extend(batch['id'])
            candi_sum.extend(batch['candi_sum_field'].squeeze(-1).long().cpu().numpy().tolist())

            sorted_batch, sorted_idx = torch.sort(out_dict['logits'], dim=-1, descending=True)
            label_mat = batch['label'].repeat(1, out_dict['logits'].shape[-1]).long().cuda()
            pos_mat = label_mat.eq(sorted_idx.cuda())
            pos_tensor = pos_mat.nonzero()[:, 1].cpu().numpy().tolist()

            gd_pos.extend(pos_tensor)

    print("p@1 : ", (np.sum(np.equal(gd, preds))) / len(gd))
    for idx in range(0, 10):
        # 先检查文件是否存在，不存在则写入，存在则continue
        if input_mode == 0:
            save_path = "../data/result/gcn1/{}_{}.csv".format(k_fold, idx)
        elif input_mode == 1:
            save_path = "../data/result/gcn2/{}_{}.csv".format(k_fold, idx)
        if os.path.exists(save_path):
            continue
        else:
            pd = pandas.DataFrame({'id': ids, 'gd': gd, 'preds': preds,
                                    'gd_pos': gd_pos, 'candi_sum': candi_sum})
            pd.to_csv(save_path, index=False)
            print("save path : {}".format(save_path))
            break


def run(fold_id = 0, input_mode=0):
    start_time = time.time()
    i = fold_id

    print("[setting] : ranking by bert and hinge loss")
    print()
    print("learning rate : {}".format(model_config.learning_rate))
    print("batch size : {}".format(model_config.batch_size))
    print("max epochs : {}".format(model_config.epochs))
    bert_rank_train(i, input_mode)
    end_time = time.time()
    print("[time] : {}".format(end_time - start_time))

if __name__ == "__main__":
    fold = sys.argv[1]
    input_mode = int(sys.argv[2])
    run(fold, input_mode)