import torch

from MAD import MAD

class MAD_loss(torch.nn.Module):
    def __init__(self, MAD_class, alpha, beta, target_prop=None):
        super().__init__()
        self.DTW_ = None
        self.MAD_class = MAD_class
        self.alpha = alpha
        self.beta = beta
        self.target_prop = target_prop

    def mad(self, out_conv_source, out_conv_target, labels_source, similarity_CE, sample_source=None):
        """

        :param out_conv_source:
        :param out_conv_target:
        :param labels_source:
        :return:
        Examples:
        ---------
        """
        with torch.no_grad():
            if self.MAD_class is not True:
                if self.target_prop is not None:
                    weight_X = torch.empty(size=labels_source.shape)
                    for cl in range(0, len(sample_source)):
                        weight_X[labels_source == cl] = self.target_prop[cl] / (sample_source[cl])
                else:
                    weight_X = None
                labels_source = None
            elif self.target_prop is not None:
                weight_X = torch.empty(size=labels_source.shape)
                for cl in range(0, len(sample_source)):
                    weight_X[labels_source == cl] = self.target_prop[cl] / (sample_source[cl])
            else:
                weight_X = None
            self.weight_X = weight_X
            mad = MAD(
                      additional_cost=similarity_CE, 
                      alpha=self.alpha, 
                      beta=self.beta, 
                      first_step_DTW=False)

            mad.fit(X_source=out_conv_source.transpose(1, 2), 
                    y_source=labels_source, 
                    weights_X_source=weight_X, 
                    previous_DTW=self.DTW_)

            self.cost_OT_, self._score, _ = mad.run_for_output(X_target=out_conv_target.transpose(1, 2))
            self.OT_ = mad.OT
            self.DTW_ = mad.DTW

    def l2_torch(self, labels_source, out_conv_source, out_conv_target, loop_iteration, OT):
        global_l2_matrix = torch.zeros(size=(out_conv_source.shape[0], out_conv_target.shape[0]))
        out_conv_source_sq = out_conv_source ** 2
        out_conv_target_sq = out_conv_target ** 2
        for cl in range(0, loop_iteration):
            if loop_iteration == 1:
                idx_cl = torch.arange(0, labels_source.shape[0], 1)
            else:
                idx_cl = torch.where(labels_source == cl)

            pi_DTW = self.DTW_[cl]
            pi_DTW = torch.tensor(pi_DTW)
            C1 = torch.matmul(out_conv_source_sq[idx_cl], torch.sum(pi_DTW, dim=1)).sum(-1)
            C2 = torch.matmul(out_conv_target_sq, torch.sum(pi_DTW.T, dim=1)).sum(-1)
            C3 = torch.tensordot(torch.matmul(out_conv_source[idx_cl], pi_DTW), out_conv_target,
                                 dims=([1, 2], [1, 2]))
            C4 = C1[:, None] + C2[None, :] - 2 * C3
            global_l2_matrix[idx_cl] = C4
        l2_OT_matrix = OT * global_l2_matrix
        return l2_OT_matrix

    def forward(self, out_conv_source, out_conv_target, labels_source, similarity_CE, source_sample=None):
        """

        :param out_conv_source:
        :param out_conv_target:
        :param labels_source:
        :return:

        examples:
        ---------
        >>> source = torch.rand(size=(2000, 1, 200))
        >>> target = 10 * torch.rand(size=(2000, 1, 200))
        >>> labels = torch.zeros(size=(2000,))
        >>> mad_test = MAD_loss(num_class=1, MAD_class=True)
        >>> alpha_loss, OT, DTW, cost_OT = mad_test.loss_CNN_MAD(out_conv_source=source, out_conv_target=target, labels_source=labels)
        """
        self.mad(out_conv_source, out_conv_target, labels_source, similarity_CE, source_sample)
        
        if self.MAD_class:
            loop_iteration = torch.max(labels_source).item() + 1
        else:
            loop_iteration = 1
        alpha_loss = self.l2_torch(labels_source=labels_source, out_conv_source=out_conv_source,
                                   loop_iteration=int(loop_iteration), OT=self.OT_, out_conv_target=out_conv_target)

        length = (out_conv_source.shape[-1] + out_conv_target.shape[-1]) / 2
        beta_loss = (self.OT_ * similarity_CE).sum()
        return self.alpha * alpha_loss.sum() / length, self.beta * beta_loss, self.OT_, self.DTW_
