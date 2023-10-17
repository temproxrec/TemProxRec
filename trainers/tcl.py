import torch
import torch.nn.functional as F


def TCL(query_hids, new_batch_hidden, query_times, times, num_time_items, args, batch_size, device):
    
    query_mat = query_hids.unsqueeze(1).unsqueeze(2).repeat(1, batch_size, args.maxlen, 1) #[128, 128, 50, 128]
    query_hids_normalized = query_mat / query_mat.norm(dim=3, keepdim=True) # [128, 128, 50, 128]
    hiddens_normalized = new_batch_hidden / new_batch_hidden.norm(dim=3, keepdim=True) # [128, 128, 50, 128]

    # Calculate contrastive scores
    contrastive_scores = torch.matmul(query_hids_normalized, hiddens_normalized.transpose(2, 3)) / args.temperature #[128, 128, 50, 50]

    # Calculate the upper and lower bounds for time intervals
    upper = query_times + args.interval
    upper[upper >= num_time_items] = num_time_items
    lower = query_times - args.interval
    lower[lower <= 0] = 0

    # Create a mask to select relevant time intervals -> Create contrastive labels
    labels = torch.zeros(batch_size, batch_size, args.maxlen, dtype=torch.float, device=device)
    for i, qry_time in enumerate(query_times):
        mask = (times <= upper[i]) & (times >= lower[i]) & (times != 0)
        labels[i][mask] = 1.0
        labels[i][i, :-1] = 0.0 # intra token --> negative label
        
    # Calculate inter-contrastive loss
    inter_nll_loss = label_smoothed_nll_loss(contrastive_scores, labels)

    return inter_nll_loss


def label_smoothed_nll_loss(contrastive_scores, contrastive_labels, eps=0.0):
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.
    '''
    _, bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz*bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    
    loss = loss.view(bsz, bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss.view(bsz, -1), 1)/contrastive_labels.view(bsz, -1).sum(axis=1)

    return loss.mean()