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
    testing=False,
    seed=1,
    batch_size=16,
    learning_rate=0.00002,
    #learning_rate=0.0001,
    epochs=10,
    patience=2,
    hidden_sz=192,
    max_seq_len=40,  # necessary to limit memory usage
    max_vocab_size=10000000,
    max_candi_num = 10,
    glove_file_path = "~/ML_work/glove/glove.6B.300d.txt"
)

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField

with open (model_config.glove_file_path) as fp:
    text = fp.readlines()

token_idx_map = {"tokens": dict([(line.split(' ')[0], idx) for idx, line in enumerate(text)])}

class ListWiseDatasetReader(DatasetReader):
    def __init__(self, input_mode = 0, max_seq_len=model_config.max_seq_len):
        super().__init__(lazy=False)

        self.word_tokenizer = WordTokenizer()

        # 注意：如果使用840B的GloVe，他的词未小写化，需要调整参数
        self.token_indexer = SingleIdTokenIndexer(
            namespace = 'tokens',
            lowercase_tokens = True,
        )
        self.input_mode = input_mode
        print(type(self.input_mode))
        self.token_indexers = {"tokens": self.token_indexer}
        self.max_seq_len = max_seq_len

    def text_to_instances(self, left_input_tokens,
                right_input_tokens,
                label):
        # TODO:改写这个函数
        # all_concerned_input_tokens
        left_token_field = TextField(left_input_tokens, self.token_indexers)
        fields = {"left_input_tokens_field": left_token_field}

        right_token_field = TextField(right_input_tokens, self.token_indexers)
        fields = {"right_input_tokens_field": right_token_field}

        label_field = ArrayField(array=np.array([label]))
        fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path) -> Iterator[Instance]:
        # TODO:读取数据
        data = load_jsonl(file_path)
        suffix = file_path.split("_")[-1].split('.')[0]

        label_maps = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        for data_idx, item in enumerate(data):
            if (data_idx > 1000):
                break
            # 将claim fact、clue fact、true candi、false candi都读进来
            left_sent = item['sentence1']
            right_sent = item['sentence2']


            label = label_maps[item["gold_label"]]


            left_input_tokens = [Token(x) for x in self.tokenizer(str(left_sent))]
            right_input_tokens = [Token(x) for x in self.tokenizer(str(right_sent))]

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



from torch import nn
from allennlp.data.vocabulary import Vocabulary
from collections import OrderedDict
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_log_softmax, masked_softmax, masked_mean
from torch.nn.functional import nll_loss, hinge_embedding_loss, pad

class EnhancedRCNNModel(Model):
    # 这里默认使用bert_word_embedder
    # 此模型基于bert，由于是直接获取CLS向量作为句子向量表示，因此和ELMO等注重词建模的模型很不同
    # 所以此模型中的interaction与CDR_elmo模型代码不复用
    def __init__(self, word_embed: TextFieldEmbedder):
        # 针对每个输入text passage，做embedding，或者输入CDR之前就已经用bert转换成embedding vectors
        super(EnhancedRCNNModel, self).__init__()
        vocab = Vocabulary(counter=token_idx_map)
        self.use_hand_fea = True
        self.use_global = True
        self.embedder = word_embed


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

        # sent 1
        left_input_mask = get_text_field_mask(left_input_tokens_field)
        left_input_embeddings = self.embedder(left_input_tokens_field, num_wrapping_dims=1)
        left_input_vectors = self.encoder(left_input_embeddings, left_input_mask)
        device_id = left_input_vectors.get_device()

        # sent 2
        right_input_mask = get_text_field_mask(right_input_tokens_field)
        right_input_embeddings = self.embedder(right_input_tokens_field, num_wrapping_dims=1)
        right_input_vectors = self.encoder(right_input_embeddings, right_input_mask)
        device_id = right_input_vectors.get_device()

        print(right_input_tokens_field)
        print(right_input_mask)
        print(right_input_vectors.shape)
        print(device_id)
        exit(0)
        '''
        print()
        print(input_mask.shape)
        print(input_embeddings.shape)
        print(input_vectors.shape)
        print(clues_sum_field)
        print(candi_sum_field)
        print(in_out_field)
        '''


        fc_out = []
        softmax_mask = []
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
from allennlp.common import Params
from allennlp.modules.token_embedders import Embedding

def EnhancedRCNN_train(k_fold = 0, input_mode=0):

    data_fold = k_fold
    print("process data fold {}".format(data_fold))

    vocab = Vocabulary(counter = token_idx_map)

    EMBEDDING_DIM = 300
    token_embedding = Embedding.from_params(
        vocab=vocab,
        params=Params({'pretrained_file': model_config.glove_file_path,
                       'embedding_dim': EMBEDDING_DIM})
    )
    word_embeddings = TextFieldEmbedder({"token": token_embedding})
    model = EnhancedRCNNModel(word_embeddings)

    if torch.cuda.is_available():
        cuda_device = list(range(torch.cuda.device_count()))

        model = model.cuda(cuda_device[0])
    else:
        cuda_device = -1
    print("cuda device : {}".format(cuda_device))

    reader = ListWiseDatasetReader(input_mode=input_mode)
    train_dataset = reader.read("~/2019/fever_plus/src/data/SNLI/snli/snli_1.0/snli_1.0_train.jsonl")
    dev_dataset = reader.read("~/2019/fever_plus/src/data/SNLI/snli/snli_1.0/snli_1.0_dev.jsonl")
    test_dataset = reader.read("~/2019/fever_plus/src/data/SNLI/snli/snli_1.0/snli_1.0_test.jsonl")


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
    EnhancedRCNN_train(i, input_mode)
    end_time = time.time()
    print("[time] : {}".format(end_time - start_time))

if __name__ == "__main__":
    fold = sys.argv[1]
    input_mode = int(sys.argv[2])
    run(fold, input_mode)