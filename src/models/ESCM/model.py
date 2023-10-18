### TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
    MultiHeadGCN,
)
from src.utils import config
from src.utils.constants import MAP_EMO
from sklearn.metrics import accuracy_score


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask, parents=None):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask, parents=parents)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask, parents=parents)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
        kv_input_depth=None,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
            kv_input_depth,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq

            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            return logit
        else:
            return nn.Softmax(dim=-1)(logit)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x

class ESCM(nn.Module):
    def __init__(
        self,
        vocab,
        decoder_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
        dep_tree_vocab=None,
    ):
        super(ESCM, self).__init__()
        self.emo_embed_size = 32 
        self.compress_dim = 10 
        self.cos_enc_size = self.compress_dim + self.emo_embed_size

        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.dep_tree_vocab = dep_tree_vocab
        self.dep_tree_dim = config.dep_dim

        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
        num_emotions = 32

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.tag_rel_embedding = share_embedding(self.dep_tree_vocab, pretrain=False, emb_dim=self.dep_tree_dim)

        self.encoder = self.make_encoder(config.emb_dim)
        self.emo_encoder = self.make_encoder(self.cos_enc_size, hidden_dim=self.cos_enc_size)
        self.gcn_encoder = MultiHeadGCN(
            kv_input_depth=config.emb_dim + self.dep_tree_dim * 2, 
            input_depth=self.cos_enc_size,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            output_depth=self.cos_enc_size, 
            num_heads=1,
            dropout=0.0,
        )

        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            kv_input_depth=config.hidden_dim + self.cos_enc_size
        )

        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}
        self.emotion_index = self.word2tensor(self.map_emo)
        self.emotion_linear = nn.Linear(config.emb_dim, config.emb_dim)
        self.word_linear = nn.Linear(config.emb_dim, config.emb_dim)
        self.cosine_linear = nn.Linear(self.emo_embed_size, self.emo_embed_size)
        self.w_word_embed = nn.Linear(config.emb_dim, self.compress_dim, bias=False)

        emo_att_dim = self.cos_enc_size
        self.attention_layer2 = nn.Linear(emo_att_dim, emo_att_dim)
        self.attention_v2 = nn.Linear(emo_att_dim, 1, bias=False)
        self.hidden_layer2 = nn.Linear(emo_att_dim, emo_att_dim)
        self.output_layer2 = nn.Linear(emo_att_dim, num_emotions)

        emo_att_dim = config.hidden_dim
        self.attention_layer = nn.Linear(emo_att_dim, emo_att_dim)
        self.attention_v = nn.Linear(emo_att_dim, 1, bias=False)
        self.hidden_layer = nn.Linear(emo_att_dim, emo_att_dim)
        self.output_layer = nn.Linear(emo_att_dim, num_emotions)

        if not config.woCOG:
            self.cog_lin = MLP()

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        if not config.woDiv:
            self.criterion.weight = torch.ones(self.vocab_size)

        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def make_encoder(self, emb_dim, hidden_dim=config.hidden_dim):
        return Encoder(
            emb_dim,
            hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def word2tensor(self, map_):
        list_ = []
        for key in map_:
            word = map_[key]
            index = self.vocab.word2index[word]
            list_.append(index)
        tensor = torch.from_numpy(np.array(list_)).cuda()
        return tensor

    def save_model(self, running_avg_ppl, iter, acc_val):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "ESCM_{}_{:.4f}_{:.4f}".format(iter, running_avg_ppl, acc_val),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    # Emotion Attention 
    def calc_emo_aggregation(self, enc_outputs):
        projected = self.attention_layer2(enc_outputs)
        projected = nn.Tanh()(projected)
        scores = nn.Softmax(dim=-1)(self.attention_v2(projected).squeeze(2))
        scores = scores.unsqueeze(1)  # (batch_size, 1, seq_len)
        hidden_x = torch.bmm(scores, enc_outputs).squeeze(1)
        x = self.hidden_layer2(hidden_x)
        x = nn.Tanh()(x)
        emo_logits = self.output_layer2(x)
        return emo_logits 

    # Emotion Attention 
    def calc_enc_aggregation(self, enc_outputs):
        projected = self.attention_layer(enc_outputs) #bz, src_len, emb_dim
        projected = nn.Tanh()(projected)
        scores = nn.Softmax(dim=-1)(self.attention_v(projected).squeeze(2))
        scores = scores.unsqueeze(1)  # (batch_size, 1, seq_len)
        hidden_x = torch.bmm(scores, enc_outputs).squeeze(1)
        x = self.hidden_layer(hidden_x)
        x = nn.Tanh()(x)
        emo_logits = self.output_layer(x) # bz, emotion_num
        return emo_logits 

    def gcn_layer(self, enc_emb, tag_batch, adjacency_matrix, relation_matrix, mask):
        src_len = enc_emb.shape[1]
        queries = enc_emb.unsqueeze(2).repeat(1, 1, src_len, 1)
        qtag_batch = tag_batch.unsqueeze(2).repeat(1, 1, src_len)

        keys = enc_emb.unsqueeze(1).repeat(1, src_len, 1, 1)
        ktag_batch = tag_batch.unsqueeze(1).repeat(1, src_len, 1)

        relation_emb = self.tag_rel_embedding(relation_matrix) # bz, src_len, src_len, dim
        qtag_emb = self.tag_rel_embedding(qtag_batch)
        ktag_emb = self.tag_rel_embedding(ktag_batch)

        query = torch.cat([queries, relation_emb, qtag_emb], dim=-1)
        key = torch.cat([keys, relation_emb, ktag_emb], dim=-1)
        value = enc_emb

        gcn_outputs, gcn_attentions = self.gcn_encoder(query, key, value, mask, adjacency_matrix)
        return gcn_outputs, gcn_attentions 

    def compute_dynamic_es(self, enc_batch, enc_emb):
        word_embed = self.embedding(enc_batch)
        word_embed = self.word_linear(word_embed)

        emotion_embed = self.embedding(self.emotion_index)
        emotion_embed = self.emotion_linear(emotion_embed)
        dot_score = torch.matmul(word_embed, emotion_embed.T)

        dot_score = self.cosine_linear(dot_score)
        word_embed_compress = self.w_word_embed(enc_emb)
        dynamic_es = torch.cat((word_embed_compress, dot_score), dim=2)
        return dynamic_es

    def forward(self, batch):
        ## Encode the context (Semantic Knowledge)
        enc_batch = batch["input_batch"]
        enc_parents = batch["enc_parents"]

        tag_batch = batch["tag_batch"]
        relation_matrix = batch["relation_matrix"]
        
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)

        enc_cos_embed = self.compute_dynamic_es(enc_batch, enc_outputs)
        cos_outputs = self.emo_encoder(enc_cos_embed, src_mask, enc_parents)
        adjacency_matrix = relation_matrix.data.eq(config.PAD_idx)
        gcn_outputs, gcn_att = self.gcn_layer(cos_outputs, tag_batch, adjacency_matrix, relation_matrix, src_mask)
        emo_outputs = torch.cat((enc_outputs, gcn_outputs), dim=-1)

        # Emotion logit
        es_emo_logits = self.calc_emo_aggregation(gcn_outputs)
        enc_emo_logits = self.calc_enc_aggregation(enc_outputs)
        emo_logits = es_emo_logits + enc_emo_logits 
        return src_mask, emo_outputs, emo_logits, es_emo_logits, enc_emo_logits 

    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        src_mask, ctx_output, emo_logits, es_emo_logits, enc_emo_logits = self.forward(batch)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        dec_emb = self.embedding(dec_batch_shift)
        pre_logit, attn_dist = self.decoder(dec_emb, ctx_output, (src_mask, mask_trg))

        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )
        logit = torch.log(logit)

        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        es_emo_loss = nn.CrossEntropyLoss(reduction='mean')(es_emo_logits, emo_label).to(config.device)
        enc_emo_loss = nn.CrossEntropyLoss(reduction='mean')(enc_emo_logits, emo_label).to(config.device)

        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )

        loss = ctx_loss + es_emo_loss + enc_emo_loss
        emo_loss = es_emo_loss + enc_emo_loss
        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        top_preds = ""
        comet_res = {}

        ctx_loss_list = []
        batch_size = emo_logits.size(0)
        top_preds = [[] for _ in range(batch_size)] 
        comet_res = [] 

        if self.is_eval:
            # Comet outputs
            top_probs, top_indices = emo_logits.detach().cpu().topk(3, dim=-1)
            for i, indices in enumerate(top_indices):
                top_preds[i] = [MAP_EMO[index.item()] for index in indices]
            for i in range(batch_size):
                temp_dict = {}
                for r in self.rels:
                    txt = [[" ".join(t) for t in tm] for tm in batch[f"{r}_txt"]][i]
                    temp_dict[r] = txt
                comet_res.append(temp_dict)

            # Update test batch
            for i in range(logit.shape[0]):
                logit_i = logit[i:i + 1].contiguous().view(-1, logit.size(-1))
                dec_batch_i = dec_batch[i:i + 1].contiguous().view(-1)
                loss_i = self.criterion_ppl(logit_i, dec_batch_i)
                ctx_loss_list.append(loss_i.item())

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item() if train else ctx_loss_list,
            math.exp(min(ctx_loss.item(), 100)) if train else np.mean(ctx_loss_list),  # Modify testBatch
            emo_loss.item(),
            program_acc,
            top_preds,
            comet_res,
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss


    def decoder_greedy(self, batch, max_dec_step=30):
        (   
            _,  
            _,  
            _,  
            enc_batch_extend_vocab,
            extra_zeros,
            _,  
            _,  
            _,  
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, emo_logits, es_emo_logits, enc_emo_logits = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1): 
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )   
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )   

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )   
            prob = torch.log(prob)

            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [   
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]   
            )   
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " " 
            sent.append(st)
        return sent

    def decoder_greedy_batch(self, batch, max_dec_step=30):
        (   
            _,  
            _,  
            _,  
            enc_batch_extend_vocab,
            extra_zeros,
            _,  
            _,  
            _,  
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, emo_logits, es_emo_logits, enc_emo_logits = self.forward(batch)
        batch_size = ctx_output.size(0)
        ys = torch.ones(batch_size, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []

        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )   
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )   

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None

            )   
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [   
                    "<EOS>" if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]   
            )   
            next_word = next_word.data

            ys = torch.cat(
                [ys, next_word.unsqueeze(1).long().to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sents = []
        for _, row in enumerate(np.transpose(decoded_words)):
            sent = []
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " " 
            sent.append(st)
            sents.append(sent)

        return sents

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent
