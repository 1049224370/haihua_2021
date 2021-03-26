import transformers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == False, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class modelMy(torch.nn.Module):
    def __init__(self, args, device):
        super(modelMy, self).__init__()
        self.hidden_size = 768
        self.device = device
        # self.attn_head = args.heads
        self.paper_encoder = transformers.models.gpt2.GPT2Model.from_pretrained(args.pretrained_model)

        self.Q_P_attention = MultiHeadAttention(heads = 4, d_model = self.hidden_size, dropout=0.)
        self.attn = MultiHeadAttention(4, self.hidden_size, dropout=0.)

        self.attn2 = MultiHeadAttention(4, self.hidden_size, dropout=0.)
        self.add_pe = PositionalEncoding(self.hidden_size, 0.)

        self.sent_layer_norm = nn.LayerNorm(self.hidden_size)
        self.word_layer_norm = nn.LayerNorm(self.hidden_size)
        self.query_word_layer_norm = nn.LayerNorm(self.hidden_size)
        self.query_sent_layer_norm = nn.LayerNorm(self.hidden_size)

        ## Context Wise Encoder
        self.slice_transformer = Encoder(EncoderLayer(self.hidden_size,
                                           MultiHeadAttention(4, self.hidden_size, dropout=0.1),
                                           PositionwiseFeedForward(self.hidden_size, self.hidden_size, 0.1),
                                           0.1),
                                      N=1)
        self.fuck = nn.Linear(self.hidden_size,self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.metric = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        self.nll = nn.CrossEntropyLoss(ignore_index=-1)
        self.eval_nll = nn.CrossEntropyLoss(ignore_index=-1)

    def make_aux_tensors(self, ids):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self,inputs,questions, choices, labels = None,training = True):
        seq_len = inputs.shape[-1]
        n_slices = inputs.shape[1] # n_slices
        n_questions = questions.shape[1]
        n_batch = inputs.shape[0]
        token_type_ids, attention_mask1 = self.make_aux_tensors(inputs)
        token_type_ids_q, attention_maskq = self.make_aux_tensors(questions)
        token_type_ids_c, attention_maskc = self.make_aux_tensors(choices)
        hidden = self.paper_encoder.forward(input_ids=inputs.view(-1,seq_len),
                                            token_type_ids=token_type_ids.view(-1,seq_len),
                                            attention_mask=attention_mask1.view(-1,seq_len)).last_hidden_state
        hidden_question = self.paper_encoder.forward(input_ids=questions.view(-1,seq_len),
                                                       token_type_ids=token_type_ids_q.view(-1, seq_len),
                                                       attention_mask=attention_maskq.view(-1, seq_len)).last_hidden_state
        hidden_question = hidden_question[:,0]
        hidden_choices = self.paper_encoder.forward(input_ids=choices.view(-1,256),
                                                       token_type_ids=token_type_ids_c.view(-1,256),
                                                       attention_mask=attention_maskc.view(-1,256)).last_hidden_state[:,0]
        hidden_choices  = hidden_choices.view(n_batch,n_questions,4,-1)
        hidden = torch.mul(hidden, attention_mask1.view(-1, seq_len, 1).expand(hidden.size()).float())
        hidden = hidden.repeat(n_questions, 1, 1)
        hidden_question =hidden_question.repeat(1,n_batch*n_slices).view(n_batch*n_slices*n_questions, -1)
        hidden = self.attn(self.query_word_layer_norm(hidden_question), hidden, hidden,
                           mask=attention_mask1.view(-1, 1, seq_len).repeat(n_questions,1, 1))
        
        hidden = hidden.view(n_questions, n_batch, n_slices, self.hidden_size).view(-1,n_slices,self.hidden_size)
        hidden = self.word_layer_norm(hidden)
        hidden = self.add_pe(hidden)
        slice_mask = torch.tril(torch.ones(n_slices, n_slices)).unsqueeze(0).repeat(n_questions,1,1).to(self.device)
        hidden = self.slice_transformer(hidden,slice_mask)
        hidden = self.attn2(self.query_sent_layer_norm(hidden_question).view(-1,n_slices,self.hidden_size), hidden, hidden,
                            mask=slice_mask)
        hidden = self.sent_layer_norm(self.fuck(self.dropout(hidden)))
        hidden = hidden[:,-1].unsqueeze(0).unsqueeze(2).repeat(1,1,4,1)
        _dist = -self.metric(hidden.view(-1,self.hidden_size),hidden_choices.view(-1,self.hidden_size)).view(1,n_questions,-1)
        _, pred = torch.max(_dist, -1)
        if labels is not None:
            acc = torch.mean(pred.eq(labels).view(-1).float())
            if(training == False):
                _loss = self.eval_nll(_dist.view(n_batch * n_questions, -1), labels.view(-1))
            else:
                _loss = self.nll(_dist.view(n_batch * n_questions, -1), labels.view(-1))
            return _loss,pred,acc
        return pred

    def save_pretrained(self, dir, optimizer, epoch):
        state = {'net': self.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, dir)

    def load_pretrained(self, dir):
        state_dict = torch.load(open(dir,"rb"))
        self.load_state_dict(state_dict['net'])
