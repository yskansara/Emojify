import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, batch_size, seq_len=None, evaluation=False):
    sen = source['sentences'][i: i + batch_size]
    sen = [torch.Tensor(x).view(-1, 1) for x in sen]
    targets = torch.Tensor(source['targets'][i: i + batch_size])
    lengths = source['lengths'][i: i + batch_size]
    max_len = max(lengths)
    batch = pad_sequence(sen)
    return batch.cuda(), torch.Tensor(lengths).cuda(), max_len, targets.cuda()
