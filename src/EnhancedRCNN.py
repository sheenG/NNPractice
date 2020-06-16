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

file_dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_dir_path, "../")

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
    testing=False,
    seed=1,
    batch_size=32,
    learning_rate=0.005,
    #learning_rate=0.0001,
    epochs=10,
    patience=2,
    hidden_sz=192,
    max_seq_len=40,  # necessary to limit memory usage
    min_seq_len = 5,
    max_vocab_size=10000000,
    num_class = 3,
    glove_file_path = "/home/LAB/gengxin/ML_work/glove/glove.6B.300d.txt",
    save_path = "result/result.csv",
    snli_base_path = "/home/LAB/gengxin/2019/fever_plus/src/data/SNLI/snli/snli_1.0/"
)

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField
from allennlp.data.vocabulary import Vocabulary



class ListWiseDatasetReader(DatasetReader):
    def __init__(self, max_seq_len=model_config.max_seq_len, min_seq_len = model_config.min_seq_len, vocab = None):
        super().__init__(lazy=False)

        self.word_tokenizer = WordTokenizer()
        self.vocab = vocab

        # 注意：如果使用840B的GloVe，他的词未小写化，需要调整参数
        self.token_indexer = SingleIdTokenIndexer(
            namespace = 'tokens',
            lowercase_tokens = False,
        )
        self.token_indexers = {"tokens": self.token_indexer}
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

    def text_to_instances(self, left_input_tokens,
                right_input_tokens,
                label):
        # TODO:改写这个函数
        # all_concerned_input_tokens
        left_token_field = TextField(left_input_tokens, self.token_indexers)
        fields = {"left_input_tokens_field": left_token_field}

        right_token_field = TextField(right_input_tokens, self.token_indexers)
        fields["right_input_tokens_field"] = right_token_field

        label_field = ArrayField(array=np.array([label]))
        fields["label"] = label_field

        '''
        for token in left_input_tokens:
            print(token.text, sep=' ')
            print(token.text_id)

        print(self.token_indexer.tokens_to_indices(left_input_tokens, self.vocab, "tokens"))
        print(self.token_indexer.tokens_to_indices(right_input_tokens, self.vocab, "tokens"))
        exit(0)
        '''

        return Instance(fields)

    def _read(self, file_path) -> Iterator[Instance]:
        # TODO:读取数据
        data = load_jsonl(file_path)
        #suffix = file_path.split("_")[-1].split('.')[0]

        label_maps = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        data_amount_limit_map = {'train': 600000, 'dev': 100000, 'test': 100000}
        data_limit = 0
        for key, v in data_amount_limit_map.items():
            if (file_path.find(key) != -1):
                data_limit = data_amount_limit_map[key]
                break
        for data_idx, item in enumerate(data):
            if (data_idx > data_limit):
                break
            # 将claim fact、clue fact、true candi、false candi都读进来
            left_sent = item['sentence1']
            right_sent = item['sentence2']


            if item["gold_label"] not in label_maps.keys():
                continue

            label = label_maps[item["gold_label"]]

            left_input_tokens = self.tokenizer(str(left_sent))
            right_input_tokens = self.tokenizer(str(right_sent))

            if (len(left_input_tokens) < self.min_seq_len or len(right_input_tokens) < self.min_seq_len):
                continue

            yield self.text_to_instances(
                left_input_tokens,
                right_input_tokens,
                label
            )

    def tokenizer(self, sent: str):
        # 这里使用word_piece形式切分，得到的token 粒度更小
        return self.word_tokenizer.tokenize(sent)

from allennlp.models import Model
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import move_to_device



from torch import nn
from collections import OrderedDict
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_log_softmax, masked_softmax, masked_mean
from torch.nn.functional import nll_loss, hinge_embedding_loss, pad, relu, softmax, log_softmax
from torch import tanh
from torch.nn import GRU
from torch.nn import Conv1d


class EnhancedRCNNModel(Model):
    # 这里默认使用bert_word_embedder
    # 此模型基于bert，由于是直接获取CLS向量作为句子向量表示，因此和ELMO等注重词建模的模型很不同
    # 所以此模型中的interaction与CDR_elmo模型代码不复用
    def __init__(self, word_embed: TextFieldEmbedder, num_class: int, vocab):
        # 针对每个输入text passage，做embedding，或者输入CDR之前就已经用bert转换成embedding vectors
        super(EnhancedRCNNModel, self).__init__(vocab)
        self.vocab = vocab
        self.use_hand_fea = True
        self.use_global = True
        self.embedder = word_embed

        # Bi-LSTM encoder, hidden size 192 follow paper
        self.GRU_encoder = GRU(input_size=300, hidden_size=192, num_layers=1, bias=True, batch_first=True,
                                dropout=0, bidirectional=True)

        # 这里运用卷积时，in_channels指的是前一步rnn的步长（seq_len），forward的时候应当把rnn输出维度顺序调整一下
        # 第二个参数表示实际使用到的卷积核的个数
        self.cnn_feas = 50
        self.cnn_k1_1 = Conv1d(in_channels=192 * 2, out_channels=self.cnn_feas, kernel_size=1)
        self.cnn_k1_2 = Conv1d(in_channels=192 * 2, out_channels=self.cnn_feas, kernel_size=1)
        self.cnn_k1_3 = Conv1d(in_channels=192 * 2, out_channels=self.cnn_feas, kernel_size=1)

        self.cnn_k2_1 = Conv1d(in_channels=self.cnn_feas, out_channels=self.cnn_feas, kernel_size=2)
        self.cnn_k3_1 = Conv1d(in_channels=self.cnn_feas, out_channels=self.cnn_feas, kernel_size=3)

        self.linear1 = nn.Linear((self.cnn_feas * 6 + (192 * 2) * 4 * 2) * 4,
                                 (self.cnn_feas * 6 + (192 * 2) * 4 * 2))

        self.gating = nn.Sequential(
            nn.Linear(2 * (self.cnn_feas * 6 + (192 * 2) * 4 * 2), self.cnn_feas * 6 + (192 * 2) * 4 * 2),
            nn.Sigmoid()
        )

        self.linear2 = nn.Linear((self.cnn_feas * 6 + (192 * 2) * 4 * 2) * 2, num_class)
        '''
        self.bert_prediction = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(bert_sent_len + 100 + 8, 1)),
            ('activation', nn.ReLU(inplace=True)),
        ]))
        '''

        self.accuracy = CategoricalAccuracy()
        #self.loss_type = "m-hinge-loss"
        self.loss_type = "CE"
        pass

    def forward(self, left_input_tokens_field, right_input_tokens_field, label):
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

        '''
        print(type(left_input_tokens_field))
        print(self.vocab.get_token_index('apple'))
        print(list(self.vocab._token_to_index['tokens'].items())[:15])
        '''

        # sent 1
        #left_input_mask = get_text_field_mask(left_input_tokens_field)
        left_input_embeddings = self.embedder(left_input_tokens_field)

        # sent 2
        #right_input_mask = get_text_field_mask(right_input_tokens_field)
        right_input_embeddings = self.embedder(right_input_tokens_field)

        left_rnn_out, _ = self.GRU_encoder(left_input_embeddings)
        right_rnn_out, _ = self.GRU_encoder(right_input_embeddings)
        device_id = right_rnn_out.get_device()

        # rnn结果输入到cnn之前，置换一下维度，满足nn.conv的要求
        permuted_left_rnn_out = left_rnn_out.permute(0, 2, 1)
        permuted_right_rnn_out = right_rnn_out.permute(0, 2, 1)

        left_cnn_k1_out1 = relu(self.cnn_k1_1(permuted_left_rnn_out))
        left_cnn_k1_out2 = relu(self.cnn_k1_2(permuted_left_rnn_out))
        left_cnn_k1_out3 = relu(self.cnn_k1_3(permuted_left_rnn_out))

        right_cnn_k1_out1 = relu(self.cnn_k1_1(permuted_right_rnn_out))
        right_cnn_k1_out2 = relu(self.cnn_k1_2(permuted_right_rnn_out))
        right_cnn_k1_out3 = relu(self.cnn_k1_3(permuted_right_rnn_out))

        left_cnn_k2_out = relu(self.cnn_k2_1(left_cnn_k1_out2))
        left_cnn_k3_out = relu(self.cnn_k3_1(left_cnn_k1_out3))

        right_cnn_k2_out = relu(self.cnn_k2_1(right_cnn_k1_out2))
        right_cnn_k3_out = relu(self.cnn_k3_1(right_cnn_k1_out3))

        '''
        print(right_cnn_k1_out1.shape)
        print(right_cnn_k2_out.shape)
        print(right_cnn_k3_out.shape)
        exit(0)
        '''

        # 这里关于CNN的结构参考EnhancedRCNN文中结构图
        left_cnn_k1_ave = torch.mean(left_cnn_k1_out1, dim=2)
        left_cnn_k2_ave = torch.mean(left_cnn_k2_out, dim=2)
        left_cnn_k3_ave = torch.mean(left_cnn_k3_out, dim=2)

        right_cnn_k1_ave = torch.mean(right_cnn_k1_out1, dim=2)
        right_cnn_k2_ave = torch.mean(right_cnn_k2_out, dim=2)
        right_cnn_k3_ave = torch.mean(right_cnn_k3_out, dim=2)

        left_cnn_k1_max, _ = torch.max(left_cnn_k1_out1, dim=2)
        left_cnn_k2_max, _ = torch.max(left_cnn_k2_out, dim=2)
        left_cnn_k3_max, _ = torch.max(left_cnn_k3_out, dim=2)

        right_cnn_k1_max, _ = torch.max(right_cnn_k1_out1, dim=2)
        right_cnn_k2_max, _ = torch.max(right_cnn_k2_out, dim=2)
        right_cnn_k3_max, _ = torch.max(right_cnn_k3_out, dim=2)

        left_cnn_concat = torch.cat((left_cnn_k1_ave, left_cnn_k2_ave, left_cnn_k3_ave,
                                     left_cnn_k1_max, left_cnn_k2_max, left_cnn_k3_max), dim=1)
        right_cnn_concat = torch.cat((right_cnn_k1_ave, right_cnn_k2_ave, right_cnn_k3_ave,
                                     right_cnn_k1_max, right_cnn_k2_max, right_cnn_k3_max), dim=1)

        # 然后是GRU输出进行interaction
        dot_attention_score_matrix = torch.bmm(left_rnn_out, permuted_right_rnn_out)
        # score shape: [b, len1, glove_len] * [b, glove_len, len2] --> [b, len1, len2]

        att_weights1 = softmax(dot_attention_score_matrix, 1).permute(0, 2, 1)
        att_weights2 = softmax(dot_attention_score_matrix, 2)

        left_att_rep = torch.bmm(att_weights2, right_rnn_out)
        right_att_rep = torch.bmm(att_weights1, left_rnn_out)

        left_interaction = torch.cat((left_rnn_out, left_att_rep,
                                      left_rnn_out - left_att_rep,
                                      left_rnn_out * left_att_rep), dim = 2)

        #print(left_interaction.shape)

        right_interaction = torch.cat((right_rnn_out, right_att_rep,
                                      right_rnn_out - right_att_rep,
                                      right_rnn_out * right_att_rep), dim = 2)

        # cnn 和 rnn的池化都是在seq len 维度上，因此，前面cnn 池化dim填2， 这里rnn池化dim填1
        # （由pytorch中cnn rnn维度顺序决定）
        left_interaction_ave = torch.mean(left_interaction, dim=1)
        right_interaction_ave = torch.mean(right_interaction, dim=1)

        left_interaction_max, _ = torch.max(left_interaction, dim=1)
        right_interaction_max, _ = torch.max(right_interaction, dim=1)
        #print(left_interaction_ave.shape)
        #print(left_interaction_max.shape)

        left_aggregation = torch.cat((left_interaction_ave, left_cnn_concat, left_interaction_max), dim = 1)
        right_aggregation = torch.cat((right_interaction_ave, right_cnn_concat, right_interaction_max), dim = 1)
        #print(left_aggregation.shape)
        #print(right_aggregation.shape)

        combined_rep_left_based = tanh(self.linear1(torch.cat(
            (left_aggregation, right_aggregation, left_aggregation - right_aggregation,
             left_aggregation * right_aggregation), dim=1
        )))

        combined_rep_right_based = tanh(self.linear1(torch.cat(
            (right_aggregation, left_aggregation, right_aggregation - left_aggregation,
             left_aggregation * right_aggregation), dim=1
        )))

        gated_signal_1 = self.gating(torch.cat((left_aggregation, right_aggregation), dim=1))
        gated_signal_2 = self.gating(torch.cat((right_aggregation, left_aggregation), dim = 1))

        device_ones = move_to_device(torch.ones(gated_signal_1.size()), device_id)
        fused_rep_1 = gated_signal_1 * combined_rep_left_based + (device_ones - gated_signal_1) * left_aggregation
        fused_rep_2 = gated_signal_2 * combined_rep_right_based + (device_ones - gated_signal_2) * right_aggregation

        fc_out = self.linear2(torch.cat((fused_rep_1, fused_rep_2), dim=1))

        '''
        print()
        print(input_mask.shape)
        print(input_embeddings.shape)
        print(input_vectors.shape)
        print(clues_sum_field)
        print(candi_sum_field)
        print(in_out_field)
        '''

        ###########################################################################################
        #print(fc_out)
        logits = log_softmax(fc_out, dim=1)
        #print(logits)
        #exit(0)

        output = {'logits': logits}
        #print(predictions.shape)
        label = label.squeeze(-1)

        output['loss'] = nll_loss(logits, label.long())
        self.accuracy(fc_out, label)

        return output

    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}


from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.common import Params
from allennlp.modules.token_embedders import Embedding

def EnhancedRCNN_train():

    print("enter train")
    with open (model_config.glove_file_path) as fp:
        text = fp.readlines()

    # 这里如何优雅地解决这个初始counter的问题
    glove_lines = len(text)
    token_counts = {"tokens": dict([(line.split(' ')[0], glove_lines - idx + 2) for idx, line in enumerate(text)])}
    #print(list(token_counts.items())[:10])
    vocab = Vocabulary(counter=token_counts,
                        min_count={"tokens": 1},
                        #non_padded_namespaces=['tokens'],
                        pretrained_files={'tokens': model_config.glove_file_path},
                        only_include_pretrained_words=True)

    EMBEDDING_DIM = 300
    token_embedding = Embedding.from_params(
        vocab=vocab,
        params=Params({ 'trainable': False,
                        'pretrained_file': model_config.glove_file_path,
                        'embedding_dim': EMBEDDING_DIM,
                        'vocab_namespace': "tokens"})
    )

    print("GloVe loaded")
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    model = EnhancedRCNNModel(word_embeddings, model_config.num_class, vocab=vocab)

    if torch.cuda.is_available():
        cuda_device = list(range(torch.cuda.device_count()))

        model = model.cuda(cuda_device[0])
    else:
        cuda_device = -1
    print("cuda device : {}".format(cuda_device))

    reader = ListWiseDatasetReader(vocab=vocab)
    train_dataset = reader.read(os.path.join(model_config.snli_base_path, "snli_1.0_train.jsonl"))
    dev_dataset = reader.read(os.path.join(model_config.snli_base_path, "snli_1.0_dev.jsonl"))
    test_dataset = reader.read(os.path.join(model_config.snli_base_path, "snli_1.0_test.jsonl"))

    #fc_lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=model_config.learning_rate, momentum=0.9)
    '''
    optimizer = torch.optim.SGD([{'params': model.embedder.parameters()},
                                 {'params': model.fc1.parameters(), 'lr': fc_lr},
                                 {'params': model.fc2.parameters(), 'lr': fc_lr},
                                 {'params': model.proj_1.parameters(), 'lr': fc_lr},
                                 {'params': model.proj_2.parameters(), 'lr': fc_lr},
                                 {'params': model.bert_prediction.parameters(), 'lr': fc_lr},
                                 ], lr=model_config.learning_rate, momentum=0.9)
    '''
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    iterator_train = BucketIterator(batch_size=model_config.batch_size,
                                    sorting_keys=[("left_input_tokens_field", "num_tokens"),
                                                  ("right_input_tokens_field", "num_tokens")])
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
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()

    # test
    model.eval()

    preds = []
    gd = []
    gd_pos = []

    with torch.no_grad():
        iterator_test = BucketIterator(batch_size = model_config.batch_size,
                                       sorting_keys=[("left_input_tokens_field", "num_tokens"),
                                                  ("right_input_tokens_field", "num_tokens")])
        iterator_test.index_with(vocab)
        generator_test = iterator_test(test_dataset, 1, False)
        test_start_time = time.time()
        for batch in generator_test:
            batch = move_to_device(batch, cuda_device[0])
            gd.extend(batch['label'].squeeze(-1).long().cpu().numpy().tolist())
            out_dict = model(batch['left_input_tokens_field'], batch['right_input_tokens_field'],
                             batch['label'])
            batch_pred = torch.argmax(out_dict['logits'], -1).cpu().numpy()
            preds.extend(batch_pred.tolist())

            sorted_batch, sorted_idx = torch.sort(out_dict['logits'], dim=-1, descending=True)
            label_mat = batch['label'].repeat(1, out_dict['logits'].shape[-1]).long().cuda()
            pos_mat = label_mat.eq(sorted_idx.cuda())
            pos_tensor = pos_mat.nonzero()[:, 1].cpu().numpy().tolist()

            gd_pos.extend(pos_tensor)
        test_end_time = time.time()

    print("p@1 : ", (np.sum(np.equal(gd, preds))) / len(gd))
    print("[train time] : {}".format(train_end_time - train_start_time))
    print("[test time] : {}".format(test_end_time - test_start_time))
    # 先检查文件是否存在，不存在则写入，存在则continue
    save_path = os.path.join(root_path, model_config.save_path)
    if os.path.exists(save_path):
        print("save path already exists")
    else:
        pd = pandas.DataFrame({'gd': gd, 'preds': preds})
        pd.to_csv(save_path, index=False)
        print("save to path : {}".format(save_path))


def run():
    start_time = time.time()

    print("learning rate : {}".format(model_config.learning_rate))
    print("batch size : {}".format(model_config.batch_size))
    print("max epochs : {}".format(model_config.epochs))
    EnhancedRCNN_train()
    end_time = time.time()
    print("[time] : {}".format(end_time - start_time))

if __name__ == "__main__":
    #fold = sys.argv[1]
    #input_mode = int(sys.argv[2])
    run()