import torch
import torch.nn.functional as F
from constants.constants import NUM_LABELS

def entropy_loss(logits: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    labels = labels[train_mask].type(torch.LongTensor)
    return F.cross_entropy(input=logits[train_mask, :], target=labels)


def global_accuracy(logits, labels, mask):
    pred_labels = logits[mask, :].argmax(dim=1)
    return (1.0*torch.eq(pred_labels, labels[mask])).mean().item(), pred_labels


def accracy_per_label(logits, labels, mask):
    num_labels_dict = {i: (1.0*torch.eq(labels[mask], i)).sum() for i in range(NUM_LABELS)}
    pred_labels = torch.argmax(logits[mask, :], dim=1)
    return {
        i: (((1.0*torch.eq(pred_labels, i)).sum() / num).item() if num > 0 else -1)
        for i, num in num_labels_dict.items()
    }