import torch
import torch.nn as nn


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self,
                 inter_loss_weight=1.,
                 intra_loss_weight=1.,
                 tau=1.0,
                 loss_weight: float = 1.0,
                 teacher_detach: bool = True):
        super(DIST, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, preds_S, preds_T: torch.Tensor):  # noqa
        assert preds_S.ndim == 5
        if self.teacher_detach:
            preds_T = preds_T.detach()  # noqa
        num_classes = preds_S.shape[1]
        # logits_S = logits_S.transpose(1, 3).reshape(-1, num_classes)  # noqa
        # preds_T = preds_T.transpose(1, 3).reshape(-1, num_classes)  # noqa
        # logits_S = logits_S.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, num_classes)  # noqa
        preds_S = preds_S.transpose(1, 4).reshape(-1, num_classes)  # noqa
        preds_T = preds_T.transpose(1, 4).reshape(-1, num_classes)  # noqa
        preds_S = (preds_S / self.tau).softmax(dim=1)  # noqa
        preds_T = (preds_T / self.tau).softmax(dim=1)  # noqa
        inter_loss = self.tau**2 * inter_class_relation(preds_S, preds_T)
        intra_loss = self.tau**2 * intra_class_relation(preds_S, preds_T)
        loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss
        return loss
