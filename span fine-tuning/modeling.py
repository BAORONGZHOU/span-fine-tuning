from transformers import AlbertModel
from transformers import BertModel
from torch import nn
import torch.nn.functional as F
import torch
import pytorch_pretrained_bert

BERT_MODEL_MAP = {'bert': BertModel, 'albert': AlbertModel,'bert-cased': BertModel}


class PASentConfig(object):
    def __init__(self,
                 bert_model_type='albert',
                 bert_model_path='cached_model/albert-xxlarge-v2',
                 chunk_encoder='multi_cnn',
                 do_concat=True,
                 hidden_size=1024,
                 max_chunk_number=64,
                 max_chunk_len=16,
                 hidden_dropout_prob=0.1,
                 label_number=3,
                 device=torch.device("cuda"),
                 do_baseline=False,
                 dim_feedforward=1024):
        self.bert_model_type = bert_model_type
        self.bert_model_path = bert_model_path
        self.chunk_encoder = chunk_encoder
        self.do_concat = do_concat
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_chunk_number = max_chunk_number
        self.max_chunk_len = max_chunk_len
        self.label_number = label_number
        self.device = device
        self.do_baseline = do_baseline
        self.dim_feedforward=dim_feedforward



class TokenLevelFeatureEncoder(nn.Module):
    def __init__(self, config):
        super(TokenLevelFeatureEncoder, self).__init__()
        self.bert_model = BERT_MODEL_MAP[config.bert_model_type].from_pretrained(config.bert_model_path)

    def forward(self, wordpiece_input_ids, attention_mask, token_type_ids):
        token_level_feature, token_level_sentence_embedding = self.bert_model(wordpiece_input_ids,
                                                                              attention_mask=attention_mask,
                                                                              token_type_ids=token_type_ids)
        return token_level_feature, token_level_sentence_embedding


class CNN_conv1d(nn.Module):
    def __init__(self, config, filter_size=3):
        super(CNN_conv1d, self).__init__()
        self.char_dim = config.hidden_size
        self.filter_size = filter_size
        self.out_channels = self.char_dim
        self.char_cnn = nn.Conv1d(self.char_dim, self.char_dim, kernel_size=self.filter_size,
                                  padding=0)
        self.relu = nn.ReLU()
        # print("dropout:",str(config.hidden_dropout_prob))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs):
        """
        Arguments:
            inputs: [batch_size, token_len, token_dim]
            inputs: [batch_size, token_dim]
        """
        x = inputs.transpose(1, 2).contiguous()
        x = self.char_cnn(x)
        x = self.relu(x)
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        x = self.dropout(x.squeeze(-1))
        return x

class NER_CNN(nn.Module):
    def __init__(self, config, filter_size=3):
        super(NER_CNN, self).__init__()
        self.char_dim = config.hidden_size
        self.filter_size = filter_size
        self.out_channels = self.char_dim
        self.char_cnn = nn.Conv1d(self.char_dim, self.char_dim, kernel_size=self.filter_size,
                                  padding=1)
        self.relu = nn.ReLU()
        # print("dropout:",str(config.hidden_dropout_prob))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs):
        """
        Arguments:
            inputs: [batch_size, token_len, token_dim]
            inputs: [batch_size, token_dim]
        """
        x = inputs.transpose(1, 2).contiguous()
        x = self.char_cnn(x)
        x = self.relu(x)
        x = x.transpose(1, 2).contiguous()
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttentionEmbed(nn.Module):
    def __init__(self, config):
        super(SelfAttentionEmbed, self).__init__()
        self.Dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.query = nn.Parameter(torch.randn([config.hidden_size]))
        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs):
        """
        Arguments:
            inputs: [batch_size, max_chunk_number, token_embed_dim]
            inputs: [batch_size, token_dim]
        """
        h = F.tanh(self.Dense(inputs) + self.bias)
        a = F.softmax(torch.matmul(h, self.query))
        a = a.unsqueeze(2)
        output = torch.sum(torch.mul(inputs, a), 1)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output


class ChunkLevelFeatureEncoderMultiAttention(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderMultiAttention, self).__init__()
        self.self_attention_chunk = SelfAttentionEmbed(config)
        self.self_attention_sentence = SelfAttentionEmbed(config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_attention = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                            device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_attention[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                wordpiece_index += len
        feature_for_attention = feature_for_attention.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        chunk_feature = self.self_attention_chunk(feature_for_attention)
        chunk_feature = chunk_feature.view(batch_size, self.max_chunk_number, token_embed_dim)
        chunk_level_sentence_embedding = self.self_attention_sentence(chunk_feature)
        return chunk_feature, chunk_level_sentence_embedding


class ChunkLevelFeatureEncoderCnn(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderCnn, self).__init__()
        self.cnn = CNN_conv1d(config)
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        chunk_level_sentence_embedding = self.cnn(token_level_features)
        return chunk_level_sentence_embedding, chunk_level_sentence_embedding


class ChunkLevelFeatureEncoderAttentionMaxpool(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderAttentionMaxpool, self).__init__()
        self.self_attention_chunk = SelfAttentionEmbed(config)
        self.self_attention_sentence = SelfAttentionEmbed(config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_attention = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                            device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_attention[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                wordpiece_index += len
        feature_for_attention = feature_for_attention.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        chunk_feature = self.self_attention_chunk(feature_for_attention)
        chunk_feature = chunk_feature.view(batch_size, self.max_chunk_number, token_embed_dim)
        chunk_level_sentence_embedding = F.max_pool1d(chunk_feature.transpose(1, 2).contiguous(),
                                                      kernel_size=chunk_feature.size(-2))
        chunk_level_sentence_embedding = chunk_level_sentence_embedding.squeeze()
        return chunk_feature, chunk_level_sentence_embedding


class ChunkLevelFeatureEncoderCnnMaxpool(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderCnnMaxpool, self).__init__()
        self.conv1d = CNN_conv1d(config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_cnn = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                      device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_cnn[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                wordpiece_index += len
        feature_for_cnn = feature_for_cnn.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        chunk_feature = self.conv1d(feature_for_cnn)
        chunk_feature = chunk_feature.view(batch_size, self.max_chunk_number, token_embed_dim)
        chunk_level_sentence_embedding = F.max_pool1d(chunk_feature.transpose(1, 2).contiguous(),
                                                      kernel_size=chunk_feature.size(-2))
        chunk_level_sentence_embedding = chunk_level_sentence_embedding.squeeze()
        return chunk_feature, chunk_level_sentence_embedding

class ChunkLevelFeatureEncoderNERCNN(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderNERCNN, self).__init__()
        self.conv1d = NER_CNN(config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_cnn = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                            device=self.device)
        src_mask = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len],
                               device=self.device, dtype=torch.long)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_cnn[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                        src_mask[batch_index][chunk_index][index] = 1
                wordpiece_index += len
        feature_for_cnn = feature_for_cnn.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        feature_for_cnn = self.conv1d(feature_for_cnn)
        feature_for_cnn = feature_for_cnn.view((batch_size, self.max_chunk_number, self.max_chunk_len,
                                                            token_embed_dim))
        span_adapted_features = torch.zeros_like(token_level_features, device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        span_adapted_features[batch_index][wordpiece_index + index] = \
                        feature_for_cnn[batch_index][chunk_index][index]
                wordpiece_index += len
        return span_adapted_features


class ChunkLevelFeatureEncoderAttention(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderAttention, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, dim_feedforward=config.dim_feedforward)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_attention = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                            device=self.device)
        src_mask = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len],
                               device=self.device, dtype=torch.bool)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_attention[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                        src_mask[batch_index][chunk_index][index] = True
                wordpiece_index += len
        src_mask = src_mask.view(-1, self.max_chunk_len).contiguous()
        feature_for_attention = feature_for_attention.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        feature_for_attention = self.encoder(feature_for_attention.permute(1,0,2), src_key_padding_mask=src_mask.permute(0,1))
        feature_for_attention = feature_for_attention.permute(1,0,2)
        print(feature_for_attention.size())
        feature_for_attention = feature_for_attention.view((batch_size, self.max_chunk_number, self.max_chunk_len,
                                                            token_embed_dim))
        span_adapted_features = torch.zeros_like(token_level_features, device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        span_adapted_features[batch_index][wordpiece_index + index] = feature_for_attention[batch_index][chunk_index][index]
                wordpiece_index += len
        return span_adapted_features

class ChunkLevelFeatureEncoderAttentionV2(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderAttention, self).__init__()
        attention_config = pytorch_pretrained_bert.BertConfig(vocab_size_or_config_json_file=30522,hidden_size=config.hidden_size,
                                                              intermediate_size=config.dim_feedforward,num_attention_heads=8)
        self.encoder = pytorch_pretrained_bert.modeling.BertLayer(attention_config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_attention = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                            device=self.device)
        src_mask = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len],
                               device=self.device, dtype=torch.long)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_attention[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                        src_mask[batch_index][chunk_index][index] = 1
                wordpiece_index += len
        src_mask = src_mask.view(-1, self.max_chunk_len).contiguous()
        extended_attention_mask = src_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        feature_for_attention = feature_for_attention.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        feature_for_attention = self.encoder(feature_for_attention, extended_attention_mask)
        feature_for_attention = feature_for_attention.view((batch_size, self.max_chunk_number, self.max_chunk_len,
                                                            token_embed_dim))
        span_adapted_features = torch.zeros_like(token_level_features, device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        span_adapted_features[batch_index][wordpiece_index + index] = feature_for_attention[batch_index][chunk_index][index]
                wordpiece_index += len
        return span_adapted_features

class ChunkLevelFeatureEncoderAttentionV3(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderAttentionV3, self).__init__()
        attention_config = pytorch_pretrained_bert.BertConfig(vocab_size_or_config_json_file=30522,hidden_size=config.hidden_size,
                                                              intermediate_size=config.dim_feedforward,num_attention_heads=8)
        self.encoder = pytorch_pretrained_bert.modeling.BertLayer(attention_config)
        self.encoder2 = pytorch_pretrained_bert.modeling.BertLayer(attention_config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens,attention_mask):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_attention = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                            device=self.device)
        src_mask = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len],
                               device=self.device, dtype=torch.long)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_attention[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                        src_mask[batch_index][chunk_index][index] = 1
                wordpiece_index += len
        src_mask = src_mask.view(-1, self.max_chunk_len).contiguous()
        extended_attention_mask = src_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        feature_for_attention = feature_for_attention.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        feature_for_attention = self.encoder(feature_for_attention, extended_attention_mask)
        feature_for_attention = feature_for_attention.view((batch_size, self.max_chunk_number, self.max_chunk_len,
                                                            token_embed_dim))
        span_adapted_features = torch.zeros_like(token_level_features, device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        span_adapted_features[batch_index][wordpiece_index + index] = feature_for_attention[batch_index][chunk_index][index]
                wordpiece_index += len
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        span_adapted_features = self.encoder2(span_adapted_features,extended_attention_mask)
        return span_adapted_features


class ChunkLevelFeatureEncoderMultiCnn(nn.Module):
    def __init__(self, config):
        super(ChunkLevelFeatureEncoderMultiCnn, self).__init__()
        self.token_cnn = CNN_conv1d(config)
        self.chunk_cnn = CNN_conv1d(config)
        self.max_chunk_number = config.max_chunk_number
        self.max_chunk_len = config.max_chunk_len
        self.device = config.device

    def forward(self, token_level_features, chunk_lens):
        batch_size, max_piece_len, token_embed_dim = token_level_features.size()
        feature_for_cnn = torch.zeros([batch_size, self.max_chunk_number, self.max_chunk_len, token_embed_dim],
                                      device=self.device)
        for batch_index, chunk_len in enumerate(chunk_lens):
            wordpiece_index = 0
            for chunk_index, len in enumerate(chunk_len):
                truncate_len = len
                if truncate_len > self.max_chunk_len:
                    truncate_len = self.max_chunk_len
                if truncate_len != 0:
                    for index in range(truncate_len):
                        feature_for_cnn[batch_index][chunk_index][index] = token_level_features[batch_index][
                            wordpiece_index + index]
                wordpiece_index += len
        feature_for_cnn = feature_for_cnn.view(-1, self.max_chunk_len, token_embed_dim).contiguous()
        chunk_feature = self.token_cnn(feature_for_cnn)
        chunk_feature = chunk_feature.view(batch_size, self.max_chunk_number, token_embed_dim)
        chunk_level_sentence_embedding = self.chunk_cnn(chunk_feature)
        return chunk_feature, chunk_level_sentence_embedding


CHUNK_ENCODER_MAP = {'multi_attentioin': ChunkLevelFeatureEncoderMultiAttention,
                     'cnn_maxpool': ChunkLevelFeatureEncoderCnnMaxpool,
                     'attention_maxpool': ChunkLevelFeatureEncoderAttentionMaxpool,
                     'multi_cnn': ChunkLevelFeatureEncoderMultiCnn,
                     'cnn': ChunkLevelFeatureEncoderCnn}


class PASent(nn.Module):
    def __init__(self, config):
        super(PASent, self).__init__()
        self.token_encoder = TokenLevelFeatureEncoder(config)
        self.chunk_encoder = CHUNK_ENCODER_MAP[config.chunk_encoder](config)
        self.do_concat = config.do_concat
        self.do_baseline = config.do_baseline

    def forward(self, wordpiece_input_ids, attention_mask, chunk_lens, token_type_ids):
        token_level_feature, token_level_sentence_embedding = self.token_encoder(wordpiece_input_ids, attention_mask,
                                                                                 token_type_ids)
        if self.do_baseline:
            return token_level_sentence_embedding
        chunk_feature, chunk_level_sentence_embedding = self.chunk_encoder(token_level_feature, chunk_lens)
        if self.do_concat:
            if len(token_level_sentence_embedding.size()) == 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                token_level_sentence_embedding = token_level_sentence_embedding.unsqueeze(1)
            if len(chunk_level_sentence_embedding.size()) == 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                chunk_level_sentence_embedding = chunk_level_sentence_embedding.unsqueeze(0)
            sentence_embedding = torch.cat(
                (token_level_sentence_embedding, chunk_level_sentence_embedding), 1)
        else:
            sentence_embedding = chunk_level_sentence_embedding
        return sentence_embedding


class PASentForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(PASentForSequenceClassification, self).__init__()
        self.sentence_encoder = PASent(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_number = config.label_number
        if config.do_concat:
            self.classifier = nn.Linear(config.hidden_size * 2, config.label_number)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.label_number)

    def forward(self, input_ids, attention_mask, token_type_ids,
                chunk_lens, label_ids):
        sentence_embedding = self.sentence_encoder(input_ids, attention_mask, chunk_lens,
                                                   token_type_ids)
        sentence_embedding = self.dropout(sentence_embedding)
        logits = self.classifier(sentence_embedding)
        if label_ids is None:
            return logits
        else:
            if self.label_number == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
                return loss, logits
            else:
                loss_fct = nn.CrossEntropyLoss()
                if len(logits.size()) != 2:
                    print(logits.size())
                    print(label_ids.size())
                    logits = logits.unsqueeze(0)
                loss = loss_fct(logits, label_ids.view(-1))
                return loss, logits

class LaSAForNER(nn.Module):
    def __init__(self, config):
        super(LaSAForNER, self).__init__()
        self.token_encoder = TokenLevelFeatureEncoder(config)
        self.chunk_encoder = ChunkLevelFeatureEncoderAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_number = config.label_number
        self.do_concat = config.do_concat
        self.do_baseline = config.do_baseline
        self.device = config.device
        if config.do_concat:
            self.classifier = nn.Linear(config.hidden_size * 2, config.label_number)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.label_number)

    def forward(self, input_ids, attention_mask, token_type_ids,
                chunk_lens, label_ids,label_mask,valid_ids):
        token_level_features, token_level_sentence_embedding = self.token_encoder(input_ids, attention_mask,
                                                                                 token_type_ids)
        if self.do_baseline :
            LaSA_features = token_level_features
        else :
            LaSA_features = self.chunk_encoder(token_level_features = token_level_features,chunk_lens = chunk_lens)
        if self.do_concat :
            LaSA_features = torch.cat(( token_level_features,LaSA_features),2)
        LaSA_features = self.dropout(LaSA_features)
        batch_size, max_len, feat_dim = LaSA_features.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = LaSA_features[i][j]
        logits = self.classifier(valid_output)
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None

            active_loss = label_mask.view(-1) == 1
            active_logits = logits.view(-1, self.label_number)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss,active_logits,active_labels
        else:
            return logits

class LaSAForNERV2(nn.Module):
    def __init__(self, config):
        super(LaSAForNERV2, self).__init__()
        self.token_encoder = TokenLevelFeatureEncoder(config)
        self.chunk_encoder = ChunkLevelFeatureEncoderAttentionV3(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_number = config.label_number
        self.do_concat = config.do_concat
        self.do_baseline = config.do_baseline
        self.device = config.device
        if config.do_concat:
            self.classifier = nn.Linear(config.hidden_size * 2, config.label_number)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.label_number)

    def forward(self, input_ids, attention_mask, token_type_ids,
                chunk_lens, label_ids,label_mask,valid_ids):
        token_level_features, token_level_sentence_embedding = self.token_encoder(input_ids, attention_mask,
                                                                                 token_type_ids)
        if self.do_baseline :
            LaSA_features = token_level_features
        else :
            LaSA_features = self.chunk_encoder(token_level_features = token_level_features,chunk_lens = chunk_lens,attention_mask=attention_mask)
        if self.do_concat :
            LaSA_features = torch.cat(( token_level_features,LaSA_features),2)
        LaSA_features = self.dropout(LaSA_features)
        batch_size, max_len, feat_dim = LaSA_features.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = LaSA_features[i][j]
        logits = self.classifier(valid_output)
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None

            active_loss = label_mask.view(-1) == 1
            active_logits = logits.view(-1, self.label_number)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits + 1e-10, active_labels)
            return loss,active_logits,active_labels
        else:
            return logits
