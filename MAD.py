import ot
import numpy as np
import torch
import time


class MAD:
    def __init__(self, 
                 max_iter=100, 
                 first_step_DTW=False,
                 additional_cost=None, 
                 alpha=1.0, 
                 beta=0.0, 
                 seed=100,
                 save_OT_plan=False,
                 save_DTW_matrices=False,
                 saving_path="Demo_MAD"):
        """
        :param X_source: A dataset of time series of shape (n, T, d)
        :param X_target: A dataset of time series of shape (n', T', d)
        Please note that for both X_source and X_target if d=1, X_source and X_target have to be of shape (n, T, 1) and (n', T', 1)
        :param y_source: Labels attached to X_source, if None MAD is performed other wise its C-MAD
        :param weights_X_source: distribution of the mass of X_source for the transport, if None, a uniform distribution is used
        :param weights_X_target: distribution of the mass of X_target for the transport, if None, a uniform distribution is used
        :param previous_DTW: If not None, the DTW are initialized as previous_DTW
        :param additional_cost: An additional cost matrix for the transport
        :param alpha: weight of the regular MAD cost for the transport
        :param beta: weight of the additional cost for the transport
        :param seed: the random seed of your choice
        """
        torch.manual_seed(seed)
        self.alpha = alpha
        self.beta = beta
        if additional_cost is None:
            self.beta = 0
        self.max_iter = max_iter
        self.first_step_DTW = first_step_DTW
        self.add_cost = additional_cost
        if self.add_cost is not None:
            self.add_cost = self.add_cost.cpu().detach()
        self.tab_idx = []
        self.dist_OT = []
        self.DTW = []
        self.pi_DTW_path_idx = []
        self.save_OT_plan=save_OT_plan

        self.saving_names = []
        self.saving_files = []
        if self.save_OT_plan:
            self.history_OT_plan = []
            self.saving_files.append(self.history_OT_plan)
            self.saving_names.append("history_OT_plan.npy")

        self.save_DTW_matrices=save_DTW_matrices
        if self.save_DTW_matrices:
            self.history_DTW_matrices = []
            self.saving_files.append(self.history_DTW_matrices)
            self.saving_names.append("history_DTW_matrices.npy")
        self.saving_path = saving_path
        

    def init_OT_matrix(self):
        """
        Initialises the OT matrix by finding a first transport plan with cost 1 for each pair
        :return: A transport plan as initialised
        """
        cost_OT = torch.ones(size=(self.shape_X_source[0], self.shape_X_target[0]))
        OT_tilde = ot.emd(self.X_source_a_one, self.X_target_a_one, cost_OT, numItermax=10000000)
        return OT_tilde

    def init_DTW_matrix(self):
        """
        Initialises the DTW matrices in case self.previous_DTW is not None.
        Advances through the matrices from top left corner to bottom right
        corner by randomly choosing between going right, diagonal or below
        :return: a random but valid DTW matrix
        """
        DTW_matrix = torch.zeros(size=(self.shape_X_source[1], self.shape_X_target[1]))
        ts = [0, 0]
        indices_table = [[1, 0], [0, 1], [1, 1]]
        while (ts[0] != self.shape_X_source[1] - 1) or (ts[1] != self.shape_X_target[1] - 1):
            DTW_matrix[ts[0], ts[1]] = 1
            if ts[0] == self.shape_X_source[1] - 1:
                indice_moving = 1
            elif ts[1] == self.shape_X_target[1] - 1:
                indice_moving = 0
            else:
                indice_moving = torch.randint(0, 3, (1,))
            ts[0] = ts[0] + indices_table[indice_moving][0]
            ts[1] = ts[1] + indices_table[indice_moving][1]
        DTW_matrix[-1, -1] = 1
        return DTW_matrix

    def mat_cost_OT(self):
        """
        Compute the MAD cost used for transport computation
        Goes faster if d=1, that is why the function is split

        see equation 6 in the paper :

        $$ \textbf{C}_\text{OT}(\X_source, \X_source^{\prime}, \X_target, \{\GGt^{(c)}\}_c) =
        \left\{\sum_{\tsource,\ttarget} L^{i,j}_{\tsource,\ttarget} %d(x^i_\tsource, x^{\prime j}_\ttarget)
        \pi^{(y^i)}_{\tsource \ttarget} \right\}_{i, j} $$
        :return: a (n x n') matrix of cost
        """
        mat_cost = torch.zeros(size=(self.shape_X_source[0], self.shape_X_target[0]))
        if self.one_dim:
            for lab in range(0, self.num_class):
                pi_DTW = self.DTW[lab]
                C1 = torch.matmul(self.X_source_squared[lab], torch.sum(pi_DTW, dim=1))
                C2 = torch.matmul(self.X_target_squared, torch.sum(pi_DTW.T, dim=1))
                C3 = torch.matmul(torch.matmul(self.X_source[self.tab_idx[lab], :, 0], pi_DTW[:]), self.X_target[:, :, 0].T)
                res = C1[:, None] + C2[None, :] - 2 * C3
                
                mat_cost[self.tab_idx[lab]] = res
        else:
            for lab in range(0, self.num_class):
                pi_DTW = self.DTW[lab]
                C1 = torch.matmul(self.X_source_squared[lab].transpose(1, -1), torch.sum(pi_DTW, dim=1)).sum(-1)
                C2 = torch.matmul(self.X_target_squared.transpose(1, -1), torch.sum(pi_DTW.T, dim=1)).sum(-1)
                C3 = torch.tensordot(torch.matmul(self.X_source[self.tab_idx[lab]].transpose(1, -1), pi_DTW), self.X_target,
                                        dims=([1, 2], [2, 1]))
                res = C1[:, None] + C2[None, :] - 2 * C3
                mat_cost[self.tab_idx[lab]] = res
        mat_cost /= (self.shape_X_source[1] + self.shape_X_target[1]) / 2
        return mat_cost

    def mat_dist_DTW(self, y_source_it):
        """
        Compute the MAD cost used for computing the DTW
        Goes faster if d=1, that is why the function is split

        See equation 7 in the paper :

        $$\textbf{C}_\text{DTW}^c(\X_source, \X_source^{\prime}, \GGs) =
        \left\{\sum_{i \text{ s.t. } y^i=c, j} L^{i,j}_{\tsource,\ttarget} %d(x^i_\tsource , x^{\prime j}_\ttarget)
        \gamma_{ij}\right\}_{\tsource, \ttarget}$$
        :param y_source_it: for C-MAD: the class (of X_source) for which the cost is computed
                        for MAD: X_source is taken in its entirety since y_source is set to 0 for each series
        :return: a (T x T') matrix of cost
        """
        if self.one_dim:
            if y_source_it is None:
                OTc = self.OT
                X_source_c = self.X_source[:, :, 0]
            else:
                OTc = self.OT[self.tab_idx[y_source_it]]
                X_source_c = self.X_source[self.tab_idx[y_source_it], :, 0]
            C2 = torch.matmul(OTc.sum(axis=0), self.X_target_squared)
            C3 = torch.matmul(torch.matmul(X_source_c.T, OTc), self.X_target[:, :, 0])
            res = self.X_source_squared_sum[y_source_it] + C2[None, :] - 2 * C3
        else:
            if y_source_it is None:
                OTc = self.OT
                X_source_c = self.X_source
            else:
                OTc = self.OT[self.tab_idx[y_source_it]]
                X_source_c = self.X_source[self.tab_idx[y_source_it]]
            C2 = torch.matmul(OTc.sum(0), self.X_target_squared.transpose(0, 1)).sum(-1)
            C31 = torch.matmul(X_source_c.T, OTc)
            C32 = torch.tensordot(C31, self.X_target, dims=([0, 2], [2, 0]))
            res = self.X_source_squared_sum[y_source_it] + C2[None, :] - 2 * C32
        res /= (self.shape_X_source[1] + self.shape_X_target[1]) / 2
        return res

    def path2mat(self, path):
        """
        Turns a DTW path into the corresponding matrix
        :param path: a DTW path
        :return: a DTW matrix
        """
        pi_DTW = torch.zeros((self.shape_X_source[1], self.shape_X_target[1]))
        for i, j in path:
            pi_DTW[i, j] = 1
        return pi_DTW

    def stopping_criterion(self, last_pi_DTW):
        """
        The stopping criterion of the optimization problem
        The algorithm stops when all DTW matrices have not change since last iteration
        :param last_pi_DTW: The DTW matrices of last iteration
        :return: A boolean telling if it time to stop or not
        """
        stop = True
        for lab in range(0, self.num_class):
            pi_DTW = self.DTW[lab]
            last_DTW = last_pi_DTW[lab]
            if (pi_DTW != last_DTW).any():
                stop = False
        return stop

    def fit(self, X_source, y_source=None, X_valid_source=None, y_valid_source=None, weights_X_source=None, previous_DTW=None):
        """
        solves the MAD optimization problem using Block Coordinate Descent (section optimization in the paper for more details)

        see equation 5 in the paper:

        $$ \textbf{C}_\text{OT}(\X_source, \X_source^{\prime}, \X_target, \{\GGt^{(c)}\}_c) =
        \left\{\sum_{\tsource,\ttarget} L^{i,j}_{\tsource,\ttarget} %d(x^i_\tsource, x^{\prime j}_\ttarget)
            \pi^{(y^i)}_{\tsource \ttarget} \right\}_{i, j}$$
        :return:
        """

        if torch.is_tensor(X_source) is False:
            X_source = torch.tensor(X_source)
 
        self.X_source = X_source.cpu().detach()
        self.previous_DTW = previous_DTW
        self.shape_X_source = X_source.shape
        if y_source is not None:
            if torch.is_tensor(y_source) is False:
                y_source = torch.tensor(y_source)   
            y_source = y_source.type(torch.int32)
            label_count = 0
            y_source_corrected = torch.empty(size=(self.shape_X_source[0],), dtype=torch.int)
            for lab in torch.unique(y_source):
                y_source_corrected[y_source == lab] = label_count
                label_count = label_count + 1
            self.y_source = y_source_corrected
        else:
            self.y_source = torch.zeros(size=(self.shape_X_source[0],), dtype=torch.int)
        self.num_class = len(torch.unique(self.y_source))
        if weights_X_source is None:
            self.X_source_a_one = torch.ones(size=(self.shape_X_source[0],)) / self.shape_X_source[0]
        else:
            self.X_source_a_one = weights_X_source
        
        if self.shape_X_source[-1] == 1:
            self.one_dim = True
        else:
            self.one_dim = False
        self.X_source_squared = []
        self.X_source_squared_sum = []
        for lab in range(0, self.num_class):
            if lab == 0:
                self.tab_idx.append((self.y_source == 0).nonzero().squeeze())
            else:
                self.tab_idx.append(self.y_source.eq(lab).nonzero().squeeze())
            if self.one_dim:
                X_source2 = torch.square(self.X_source[self.tab_idx[lab], :, 0])
                X_source2_sum = torch.matmul(self.X_source_a_one[self.tab_idx[lab]], X_source2)
            else:
                X_source2 = torch.square(self.X_source[self.tab_idx[lab]])
                X_source2_sum = torch.matmul(self.X_source_a_one[self.tab_idx[lab]], X_source2.transpose(0, 1)).sum(-1)
            self.X_source_squared.append(X_source2)
            self.X_source_squared_sum.append(X_source2_sum[:, None])

    def run_mad(self):
        stop = False
        current_iter = 0
        all_iteration_time = []
        while stop is not True and current_iter < self.max_iter:
            t0 = time.time()
            if (current_iter != 0) or (self.first_step_DTW is False):
                Cost_OT_alpha = self.alpha * self.mat_cost_OT()
                if self.beta != 0:
                    Cost_0T_beta = self.beta * self.add_cost
                    Cost_OT = Cost_OT_alpha + Cost_0T_beta
                else:
                    Cost_OT = Cost_OT_alpha
                self.OT = ot.emd(self.X_source_a_one, self.X_target_a_one, Cost_OT, numItermax=1000000)
                score_OT = torch.sum(self.OT * Cost_OT)
                
            dtw_score = 0
            self.pi_DTW_path_idx = []
            total_cost_dtw = []
            for lab in range(0, self.num_class):
                mat_dist = self.mat_dist_DTW(lab)
                total_cost_dtw.append(mat_dist)
                Pi_DTW_path, dtw_score_prov = torch_dtw(mat_dist)
                self.pi_DTW_path_idx.append(Pi_DTW_path)
                Pi_DTW_prov = self.path2mat(Pi_DTW_path)
                self.DTW[lab] = Pi_DTW_prov
                dtw_score += dtw_score_prov
            t1 = time.time()
            if current_iter != 0:
                stop = self.stopping_criterion(last_pi_DTW)
            last_pi_DTW = self.DTW.copy()
            if ((current_iter != 0) or (self.first_step_DTW is False)):
                if self.save_OT_plan:
                    self.history_OT_plan.append(self.OT)
                if self.save_DTW_matrices:
                    self.history_DTW_matrices.append(self.DTW)
                all_iteration_time.append(t1 - t0)
            current_iter = current_iter + 1
        else:
            return Cost_OT, score_OT, all_iteration_time

    def to_onehot(self, y=None):
        """
        onehot encode a vector of labels
        :param y: a vector of labels
        :return: y onehot encoded
        """
        if y is None:
            y = self.y_source
        n_values = torch.max(y) + 1
        return np.eye(n_values)[y]

    def predict(self, X_target, weights_X_target=None, y_source=None):
        if torch.is_tensor(X_target) is False:
            X_target = torch.tensor(X_target)

        self.X_target = X_target.cpu().detach()
        self.shape_X_target = X_target.shape

        if weights_X_target is None:
            self.X_target_a_one = torch.ones(size=(self.shape_X_target[0],)) / self.shape_X_target[0]
        else:
            self.X_target_a_one = weights_X_target
        
        if self.one_dim:
            self.X_target_squared = torch.square(self.X_target[:, :, 0]).squeeze()
        else:
            self.X_target_squared = torch.square(self.X_target).squeeze()

        if y_source is None:
            y_source = self.y_source

        if torch.is_tensor(y_source) is False:
            y_source = torch.tensor(y_source)

        for lab in range(0, self.num_class):
            if self.previous_DTW is None:
                self.DTW.append(self.init_DTW_matrix())
            else:
                self.DTW.append(self.previous_DTW[lab])
        self.OT = self.init_OT_matrix()
        self.run_mad()

        yt_onehot = self.to_onehot(y=y_source)
        y_pred = np.argmax(np.dot(self.OT.T, yt_onehot), axis=1)
        return y_pred

    def evaluate(self, X_target, y_target, weights_X_target=None, y_source=None):
        """
        To evaluate the transport plan of MAD using propagation label as a classifieur
        :param train_target_label:
        :param train_source_label:
        :return: the accuracy and the prediction for the X_target dataset
        """
        if torch.is_tensor(X_target) is False:
            X_target = torch.tensor(X_target)

        self.X_target = X_target.cpu().detach()
        self.shape_X_target = X_target.shape

        if weights_X_target is None:
            self.X_target_a_one = torch.ones(size=(self.shape_X_target[0],)) / self.shape_X_target[0]
        else:
            self.X_target_a_one = weights_X_target
        
        if self.one_dim:
            self.X_target_squared = torch.square(self.X_target[:, :, 0]).squeeze()
        else:
            self.X_target_squared = torch.square(self.X_target).squeeze()

        if y_source is None:
            y_source = self.y_source

        if torch.is_tensor(y_source) is False:
            y_source = torch.tensor(y_source)
        if torch.is_tensor(y_target) is False:
            y_target = torch.tensor(y_target)

        for lab in range(0, self.num_class):
            if self.previous_DTW is None:
                self.DTW.append(self.init_DTW_matrix())
            else:
                self.DTW.append(self.previous_DTW[lab])
        self.OT = self.init_OT_matrix()                
        self.run_mad()
        yt_onehot = self.to_onehot(y=y_source)
        y_pred = np.argmax(np.dot(self.OT.T, yt_onehot), axis=1)
        accuracy = np.mean(y_pred == y_target.numpy().astype(int))
        self.acc_target.append(accuracy)
        return accuracy

    def run_for_output(self, X_target, weights_X_target=None):
        if torch.is_tensor(X_target) is False:
            X_target = torch.tensor(X_target)

        self.X_target = X_target.cpu().detach()
        self.shape_X_target = X_target.shape

        if weights_X_target is None:
            self.X_target_a_one = torch.ones(size=(self.shape_X_target[0],)) / self.shape_X_target[0]
        else:
            self.X_target_a_one = weights_X_target
        
        if self.one_dim:
            self.X_target_squared = torch.square(self.X_target[:, :, 0]).squeeze()
        else:
            self.X_target_squared = torch.square(self.X_target).squeeze()

        for lab in range(0, self.num_class):
            if self.previous_DTW is None:
                self.DTW.append(self.init_DTW_matrix())
            else:
                self.DTW.append(self.previous_DTW[lab])
        self.OT = self.init_OT_matrix()
        return self.run_mad()

    def save_stuff(self):
        if len(self.saving_files) != 0:
            for stuff in range(0, len(self.saving_filesfiles)):
                np.save(self.saving_path + self.saving_names[stuff], self.saving_files[stuff])

@torch.jit.script
def torch_acc_matrix(cost_matrix):
    l1 = cost_matrix.shape[0]
    l2 = cost_matrix.shape[1]
    cum_sum = torch.full((l1 + 1, l2 + 1), torch.inf, dtype=cost_matrix.dtype)
    cum_sum[0, 0] = 0.
    cum_sum[1:, 1:] = cost_matrix

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] += torch.min(cum_sum[[i, i + 1, i], [j + 1, j, j]])
    return cum_sum[1:, 1:]


@torch.jit.script
def _return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = acc_cost_mat[[i - 1, i - 1,  i], [j - 1, j, j - 1]]
            argmin = torch.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def torch_dtw(cost_matrix):
    acc_matrix = torch_acc_matrix(cost_matrix=cost_matrix)
    path = _return_path(acc_matrix)
    return path, acc_matrix[-1, -1]

