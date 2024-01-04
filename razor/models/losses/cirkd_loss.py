# https://github.com/winycg/CIRKD/blob/main/losses/cirkd_mini_batch.py
"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class CriterionMiniBatchCrossImagePair(nn.Module):
    def __init__(self, temperature, pooling=False, loss_weight=1.0, single_batch=True):
        super(CriterionMiniBatchCrossImagePair, self).__init__()
        self.temperature = temperature
        self.pooling = pooling
        self.single_batch = single_batch
        self.loss_weight = loss_weight

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W, D = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def forward(self, feat_S, feat_T):
        # feat_T = feat_T.detach()
        B, C, H, W, D = feat_S.size()

        if self.pooling:
            avg_pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, ceil_mode=True)
            feat_S = avg_pool(feat_S)
            feat_T = avg_pool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        sim_dis = torch.tensor(0.).cuda()
        if self.single_batch:
            B = 1
        for i in range(B):
            for j in range(B):
                s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_
        sim_dis = sim_dis / (B * B)
        return self.loss_weight * sim_dis


class StudentSegContrast(nn.Module):
    def __init__(self,
                 s_channels,
                 t_channels,
                 num_classes,
                 pixel_memory_size=20000,
                 region_memory_size=2000,
                 region_contrast_size=256,
                 pixel_contrast_size=1024,
                 contrast_kd_temperature=1.0,
                 contrast_temperature=0.1,
                 ignore_label=0,
                 pixel_weight=0.1,
                 region_weight=0.1,
                 loss_weight=1.0):
        super(StudentSegContrast, self).__init__()
        self.base_temperature = 0.1
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        self.dim = t_channels
        self.ignore_label = ignore_label
        self.n_view = 32
        self.pooling = True

        self.project_head = nn.Sequential(
            nn.Conv3d(s_channels, t_channels, 1, bias=False),
            nn.SyncBatchNorm(t_channels),
            nn.ReLU(True),
            nn.Conv3d(t_channels, t_channels, 1, bias=False)
        )

        self.num_classes = num_classes
        self.region_memory_size = region_memory_size
        self.pixel_memory_size = pixel_memory_size
        self.pixel_update_freq = 16
        self.pixel_contrast_size = pixel_contrast_size
        self.region_contrast_size = region_contrast_size

        self.register_buffer("teacher_segment_queue", torch.randn(self.num_classes, self.region_memory_size, self.dim))
        self.teacher_segment_queue = nn.functional.normalize(self.teacher_segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("teacher_pixel_queue", torch.randn(self.num_classes, self.pixel_memory_size, self.dim))
        self.teacher_pixel_queue = nn.functional.normalize(self.teacher_pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

        self.pixel_weight = pixel_weight
        self.region_weight = region_weight
        self.loss_weight = loss_weight

    def _sample_negative(self, Q, index):
        class_num, cache_size, feat_size = Q.shape
        contrast_size = index.size(0)
        X_ = torch.zeros((class_num * contrast_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * contrast_size, 1)).float().cuda()
        sample_ptr = 0

        for ii in range(class_num):
            this_q = Q[ii, index, :]
            X_[sample_ptr:sample_ptr + contrast_size, ...] = this_q
            y_[sample_ptr:sample_ptr + contrast_size, ...] = ii
            sample_ptr += contrast_size

        return X_, y_

    def _dequeue_and_enqueue(self, keys, labels):
        segment_queue = self.teacher_segment_queue
        pixel_queue = self.teacher_pixel_queue

        # keys = self.concat_all_gather(keys)
        # labels = self.concat_all_gather(labels)

        batch_size, feat_dim, H, W, D = keys.size()

        # bs = torch.randint(0, batch_size, (1,))

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.region_memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.pixel_memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + K) % self.pixel_memory_size

    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits / self.contrast_kd_temperature, dim=1)
        p_t = F.softmax(t_logits / self.contrast_kd_temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature ** 2
        return sim_dis

    def forward(self, s_feats, t_feats, labels=None, predict=None):
        if self.pooling:
            avg_pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, ceil_mode=True)
            s_feats = avg_pool(s_feats)
            t_feats = avg_pool(t_feats)
        t_feats = t_feats.detach()
        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = self.project_head(s_feats)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(
            labels,
            (s_feats.shape[2], s_feats.shape[3], s_feats.shape[4]),
            mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == s_feats.shape[-1], '{} {}'.format(labels.shape, s_feats.shape)

        ori_s_fea = s_feats
        ori_t_fea = t_feats
        ori_labels = labels

        batch_size = s_feats.shape[0]

        labels = labels.contiguous().view(-1)
        predict = predict.contiguous().view(batch_size, -1)

        idxs = (labels != self.ignore_label)
        s_feats = s_feats.permute(0, 2, 3, 4, 1)
        s_feats = s_feats.contiguous().view(-1, s_feats.shape[-1])
        s_feats = s_feats[idxs, :]

        t_feats = t_feats.permute(0, 2, 3, 4, 1)
        t_feats = t_feats.contiguous().view(-1, t_feats.shape[-1])
        t_feats = t_feats[idxs, :]

        self._dequeue_and_enqueue(ori_t_fea.detach().clone(), ori_labels.detach().clone())

        if idxs.sum() == 0:  # just a trick to skip all ignored anchor embeddings
            return 0. * (s_feats ** 2).mean() + 0. * (s_feats ** 2).mean()

        class_num, pixel_queue_size, feat_size = self.teacher_pixel_queue.shape
        perm = torch.randperm(pixel_queue_size)
        pixel_index = perm[:self.pixel_contrast_size]
        t_X_pixel_contrast, t_y_pixel_contrast = self._sample_negative(self.teacher_pixel_queue, pixel_index)

        t_pixel_logits = torch.div(torch.mm(t_feats, t_X_pixel_contrast.T), self.contrast_temperature)
        s_pixel_logits = torch.div(torch.mm(s_feats, t_X_pixel_contrast.T), self.contrast_temperature)

        class_num, region_queue_size, feat_size = self.teacher_segment_queue.shape
        perm = torch.randperm(region_queue_size)
        region_index = perm[:self.region_contrast_size]
        t_X_region_contrast, _ = self._sample_negative(self.teacher_segment_queue, region_index)

        t_region_logits = torch.div(torch.mm(t_feats, t_X_region_contrast.T), self.contrast_temperature)
        s_region_logits = torch.div(torch.mm(s_feats, t_X_region_contrast.T), self.contrast_temperature)

        pixel_sim_dis = self.contrast_sim_kd(s_pixel_logits, t_pixel_logits.detach())
        region_sim_dis = self.contrast_sim_kd(s_region_logits, t_region_logits.detach())

        return self.loss_weight * self.pixel_weight * pixel_sim_dis + self.loss_weight * self.region_weight * region_sim_dis
        # return dict(
        #     pixel_conrast_loss=self.loss_weight * self.pixel_weight * pixel_sim_dis,
        #     region_contrast_loss=self.loss_weight * self.region_weight * region_sim_dis)


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1, loss_weight=1.0):
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, pred, soft):
        soft = soft.detach()
        B, C, h, w, d = soft.size()
        scale_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        scale_soft = soft.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        p_t = F.softmax(scale_soft / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        return self.loss_weight * loss
