import os
from torch import nn
import torch
import numpy as np
from UnivariateSRBenchmark.Methods.TPSR.symbolicregression.model.model_wrapper import ModelWrapper
from UnivariateSRBenchmark.Methods.TPSR.symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
import time
from UnivariateSRBenchmark.utils import *
import torch.nn.functional as F
from UnivariateSRBenchmark.Methods.TPSR.symbolicregression.model.transformer import BeamHypotheses


def generate_beam_from_state(
        decoder, src_enc, src_len, state, beam_size, top_k, top_k_hash, use_prefix_cache, length_penalty, early_stopping,
        max_len=200,
):
    """
    Decode a sentence given current state (incomplete equation).
    `x`:
        - LongTensor(bs, slen)
            <EOS> W1 W2 W3 <EOS> <PAD>
            <EOS> W1 W2 W3   W4  <EOS>
    `lengths`:
        - LongTensor(bs) [5, 6]
    `positions`:
        - False, for regular "arange" positions (LM)
        - True, to reset positions from the new generation (MT)
    """

    # check inputs
    assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1
    # batch size / number of words
    bs = len(src_len)
    n_words = decoder.n_words

    # expand to beam size the source latent representations / source lengths
    src_enc = (
        src_enc.unsqueeze(1)
        .expand((bs, beam_size) + src_enc.shape[1:])
        .contiguous()
        .view((bs * beam_size,) + src_enc.shape[1:])
    )
    src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    state = state.transpose(0, 1)

    # generated sentences (batch with beam current hypotheses)
    generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    generated.fill_(decoder.pad_index)  # fill upcoming ouput with <PAD>
    generated[:len(state)] = state

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
        for _ in range(bs)
    ]

    # positions
    positions = src_len.new(max_len).long()
    positions = (
        torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)
    )

    # scores for each sentence in the beam
    beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = len(state)

    # cache compute states
    decoder.cache = {"slen": 0}

    # done sentences
    done = [False for _ in range(bs)]
    # breakpoint()
    while cur_len < max_len:
        # compute word scores
        tensor = decoder.forward(
            "fwd",
            x=generated[:cur_len],
            lengths=src_len.new(bs * beam_size).fill_(cur_len),
            positions=positions[:cur_len],
            causal=True,
            src_enc=src_enc,
            src_len=src_len,
            use_cache=True,
        )

        if decoder.apex:
            tensor = tensor.data[-1, :, :].to(decoder.dtype)  # (bs * beam_size, dim)
        else:
            tensor = tensor.data[
                -1, :, :
            ]  # .to(soui elf.dtype)  # (bs * beam_size, dim)
        scores = decoder.proj(tensor)  # (bs * beam_size, n_words)
        scores = F.log_softmax(scores.float(), dim=-1)  # (bs * beam_size, n_words)
        assert scores.size() == (bs * beam_size, n_words)

        # select next words with scores
        _scores = scores + beam_scores[:, None].expand_as(
            scores
        )  # (bs * beam_size, n_words)
        _scores = _scores.view(bs, beam_size * n_words)  # (bs, beam_size * n_words)

        next_scores, next_words = torch.topk(
            _scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )
        assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

        if use_prefix_cache:
            state_beam = tuple(generated[:cur_len][:, 0].tolist())
            if state_beam not in top_k_hash:
                top_k_hash[state_beam] = next_words[0].tolist()

        # next batch beam content
        # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for sent_id in range(bs):

            # if we are done with this sentence
            done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                next_scores[sent_id].max().item()
            )
            if done[sent_id]:
                next_batch_beam.extend(
                    [(0, decoder.pad_index, 0)] * beam_size
                )  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                # get beam and word IDs
                beam_id = torch.div(idx, n_words, rounding_mode="trunc")
                word_id = idx % n_words

                # end of sentence, or next word
                if word_id == decoder.eos_index or cur_len + 1 == max_len:
                    generated_hyps[sent_id].add(
                        generated[:cur_len, sent_id * beam_size + beam_id]
                        .clone()
                        .cpu(),
                        value.item(),
                    )
                else:
                    next_sent_beam.append(
                        (value, word_id, sent_id * beam_size + beam_id)
                    )

                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
            if len(next_sent_beam) == 0:
                next_sent_beam = [
                                     (0, decoder.pad_index, 0)
                                 ] * beam_size  # pad the batch
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == beam_size * (sent_id + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == bs * beam_size
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = generated.new([x[1] for x in next_batch_beam])
        beam_idx = src_len.new([x[2] for x in next_batch_beam])

        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in decoder.cache.keys():
            if k != "slen":
                decoder.cache[k] = (
                    decoder.cache[k][0][beam_idx],
                    decoder.cache[k][1][beam_idx],
                )
        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    # select the best hypotheses
    tgt_len = src_len.new(bs)
    best = []

    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    decoded = src_len.new(tgt_len.max().item(), bs).fill_(decoder.pad_index)
    for i, hypo in enumerate(best):
        decoded[: tgt_len[i] - 1, i] = hypo
        decoded[tgt_len[i] - 1, i] = decoder.eos_index

    return decoded, tgt_len, generated_hyps, top_k_hash


class Transformer(nn.Module):
    def __init__(self, params, samples):
        super().__init__()
        model_path = Path(os.path.join(get_project_root(), "UnivariateSRBenchmark\\Methods\\E2E\\model.pt"))
        self.model = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
        self.first_dropout = nn.Dropout(0.1)
        self.params = params
        self.env = self.model.env
        self.embedder, self.encoder, self.decoder = self.model.embedder, self.model.encoder, self.model.decoder
        self.samples = samples

        x_to_fit = samples['x_to_fit']
        y_to_fit = samples['y_to_fit']
        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                if np.isscalar(y_to_fit[seq_id][seq_l]):
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], np.array([y_to_fit[seq_id][seq_l]])])
                else:
                    x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
        self.x1, self.src_len = self.embedder(x1)
        self.encoded = self.encoder("fwd", x=self.x1, lengths=self.src_len, causal=False).transpose(0, 1)

    def generate_beams(self, input_ids,
                top_k,
                num_beams,
                length_penalty,
                early_stopping,
                max_length,
                top_k_hash,
                use_prefix_cache
                ):

        decoded, tgt_len, generated_hyps, top_k_hash = generate_beam_from_state(self.decoder,
                                                                                self.encoded, self.src_len, input_ids, num_beams,top_k, top_k_hash, use_prefix_cache,length_penalty, early_stopping, max_len=200)

        return generated_hyps, top_k_hash


    def top_k(self, input_ids, top_k):
        bs = len(self.src_len)
        max_len=200

        # positions
        positions = self.src_len.new(max_len).long()
        positions = (
            torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        )

        state = input_ids.transpose(0, 1)

        gen_len = self.src_len.clone().fill_(len(state))
        # cache compute states
        self.decoder.cache = {"slen": 0}

        # compute word scores
        tensor = self.decoder.forward(
            "fwd",
            x=state,
            lengths=gen_len,
            positions=positions[:len(state)],
            causal=True,
            src_enc=self.encoded,
            src_len=self.src_len,
            use_cache=True, )

        # breakpoint()
        tensor = tensor.data[-1, :, :].to(self.decoder.dtype)  # (bs, dim)
        scores = self.decoder.proj(tensor)  # (bs, n_words)

        top_k_tokens = torch.topk(scores, top_k)[1].squeeze(1)

        return top_k_tokens

    
def pred_for_sample_no_refine(model, env,preds,x_to_fit):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,)
    generations_tree = list(filter(lambda x: x is not None,
                [env.idx_to_infix(preds[1:], is_float=False, str_array=False)]))
    if generations_tree == []: 
        y = None
        model_str= None
    else:
        numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(generations_tree[0]) 
        y = numexpr_fn(x_to_fit[0])[:,0]  
        model_str = generations_tree[0].infix()
    return y, model_str , generations_tree


def pred_for_sample(model, env,x_to_fit,y_to_fit, refine=True, beam_type='search', beam_size=1):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,
            beam_type= beam_type,
            beam_size= beam_size)
    dstr = SymbolicTransformerRegressor(model=mw,n_trees_to_refine=beam_size)
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    dstr.fit(x_to_fit[0], y_to_fit[0], verbose=False)
    if refine:
        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
    else:
        tree = dstr.retrieve_tree(refinement_type="NoRef", with_infos=True)["relabed_predicted_tree"]
    model_str = tree.infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
    numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
    y = numexpr_fn(x_to_fit[0])[:,0]  
    return y, model_str , [tree]


def refine_for_sample(params, model, env, preds ,x_to_fit, y_to_fit):
    embedder = model.embedder
    encoder = model.encoder
    decoder = model.decoder
    embedder.eval()
    encoder.eval()
    decoder.eval()

    mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=200,
            beam_early_stopping=True,
            beam_temperature=1.0,
            beam_type="search",
        )
    
    dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=200,
            n_trees_to_refine=5,
        )

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    generations_tree = list(filter(lambda x: x is not None,
                [env.idx_to_infix(preds[1:], is_float=False, str_array=False)]))

    if generations_tree == []: 
        y = None
        model_str= None
        tree = None
    else:
        dstr.start_fit = time.time()
        X = x_to_fit[0]
        Y = y_to_fit[0]
        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)
        dstr.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
            X[i] = X[i][:, dstr.top_k_features[i]]
        dstr.tree = {}

        refined_candidate = dstr.refine(X[0], Y[0], generations_tree, verbose=False)
        dstr.tree[0] = refined_candidate

        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
        model_str = tree.infix()
        for op,replace_op in replace_ops.items():
            model_str = model_str.replace(op,replace_op)

        numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
        y = numexpr_fn(x_to_fit[0])[:,0]  

    return y, model_str, [tree]


def refine_for_sample_test(model, env, preds ,x_to_fit, y_to_fit,x_to_pred):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,)
    dstr = SymbolicTransformerRegressor(model=mw,)
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    generations_tree = list(filter(lambda x: x is not None,
                [env.idx_to_infix(preds[1:], is_float=False, str_array=False)]))
    if generations_tree == []: 
        y = None
        model_str= None
        tree = None
    else:
        dstr.start_fit = time.time()
        X = x_to_fit[0]
        Y = y_to_fit[0]
        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)
        dstr.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
            X[i] = X[i][:, dstr.top_k_features[i]]
        dstr.tree = {}

        refined_candidate = dstr.refine(X[0], Y[0], generations_tree, verbose=False)
        dstr.tree[0] = refined_candidate

        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
        model_str = tree.infix()
        for op,replace_op in replace_ops.items():
            model_str = model_str.replace(op,replace_op)

        numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
        y_train = numexpr_fn(x_to_fit[0])[:,0]  
        y_test = numexpr_fn(x_to_pred[0])[:,0] 

    return y_train, y_test, model_str, [tree]


def respond_to_batch(model, max_target_length=200, top_p=1.0, sample_temperature=None):
    
    generations, gen_len = model.decoder.generate(
                model.encoded,
                model.src_len,
                sample_temperature=None,
                max_len=max_target_length,)
    generations = generations.transpose(0,1)
    return generations, gen_len


def pred_for_sample_test(model, env,x_to_fit,y_to_fit, x_to_pred, refine=True, beam_type='search', beam_size=1):
    mw = ModelWrapper(
            env=env,
            embedder=model.embedder,
            encoder=model.encoder,
            decoder=model.decoder,
            beam_type= beam_type,
            beam_size= beam_size)
    dstr = SymbolicTransformerRegressor(model=mw,)

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    dstr.fit(x_to_fit[0], y_to_fit[0], verbose=False)
    if refine:
        tree = dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
    else:
        tree = dstr.retrieve_tree(refinement_type="NoRef", with_infos=True)["relabed_predicted_tree"]
    model_str = tree.infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)

    numexpr_fn = mw.env.simplifier.tree_to_numexpr_fn(tree) 
    y_train = numexpr_fn(x_to_fit[0])[:,0] 
    y_test = numexpr_fn(x_to_pred[0])[:,0] 

    return y_train, y_test, model_str , [tree]
