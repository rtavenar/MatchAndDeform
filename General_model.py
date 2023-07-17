import time
import sklearn.metrics
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn as nn
import torch
from numba import jit, prange
import numpy as np
from ot import emd

from MAD_loss import MAD_loss

class Basic_CNN(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self,

                 batchsize, 
                 feature_extractor, 
                 classifier,
                 name="Default_name", 
                 X_target=None,

                 y_target=None,
                 lr=0.001, 
                 saving=True,
                 max_iterations=3000,
                 validation_step=1000,

                 CUDA_train=False,
                 save_latent_source=False):
        super().__init__()

        self.X_target = X_target
        self.y_target = y_target
        self.CUDA_train = CUDA_train
        self.name = name
        self.lr = lr
        self.max_iterations = max_iterations
        self.validation_step = validation_step
        self.saving = saving
        self.batchsize = batchsize
        self.gen = torch.Generator()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.logSoftmax = nn.LogSoftmax(dim=1)

        for i in range(0, len(self.feature_extractor)):
            if type(self.feature_extractor[i]) == nn.modules.conv.Conv1d:
                torch.nn.init.xavier_uniform_(self.feature_extractor[i].weight)

        for i in range(0, len(self.classifier)):
            if type(self.classifier[i]) == nn.modules.linear.Linear:
                torch.nn.init.xavier_uniform_(self.classifier[i].weight)
        self.optimizer = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                           {'params': self.classifier.parameters()}],
                                          lr=lr, amsgrad=True)

        self.crossLoss = nn.CrossEntropyLoss()
        self.iteration = 0
        self.loss_count = []
        self.loss_count_valid = []
        self.acc_source = []
        self.acc_target = []
        self.save_latent_source=save_latent_source
        self.training_time_ = []
        self.saving_names = ["loss_train.npy", 
                             "loss_valid.npy", 
                             "acc_source.npy", 
                             "acc_target.npy",
                             "training_time.npy"]
        self.saving_files = [self.loss_count, 
                             self.loss_count_valid, 
                             self.acc_source,
                             self.acc_target,
                             self.training_time_]
        if self.save_latent_source:
            self.history_latent_source = []
            self.saving_files.append(self.history_latent_source)
            self.saving_names.append("history_latent_source.npy")
        if self.CUDA_train:
            if torch.cuda.is_available():
                print('Cuda available')
                self.feature_extractor = self.feature_extractor.cuda()
                self.classifier = self.classifier.cuda()
                self.crossLoss = self.crossLoss.cuda()
                self.logSoftmax = self.logSoftmax.cuda()

    def draw_batches(self, training=True):
        if training:
            self.batch_source, self.batch_labels_source, self.batch_index_source = self.mini_batch_class_balanced(X=self.X_source_,
                                                                                                                  classes_proportion=self.sample_vec,
                                                                                                                  y=self.y_source_)
        else:
            self.batch_source_valid, self.batch_labels_source_valid, self.batch_index_source_valid = self.mini_batch_class_balanced(X=self.X_source_valid,
                                                                                                                                    classes_proportion=self.sample_vec_valid,
                                                                                                                                    y=self.y_source_valid)                                                                                                

    def mini_batch_class_balanced(self, X, classes_proportion, y=None):
        """
        Draw a batch at random. If y is given then the batch is drawn with respect to the number of classes given in
        classes_proportion
        :param X: A dataset
        :param classes_proportion: the distribution of the batchsize if y is not None
        :param y: the labels correpsonding to X
        :return: A batch, the corresponding labels and their index in the whole dataset
        """
        if y is not None:
            rindex = torch.randperm(len(X))
            X = X[rindex]
            y = y[rindex]
            index = torch.tensor([])
            if self.CUDA_train:
                if torch.cuda.is_available():
                    index = index.cuda()
            for i in range(self.n_classes):
                s_index = torch.nonzero(y == i).squeeze()
                index_random = torch.randperm(n=s_index.shape[0], generator=self.gen)
                s_ind = s_index[index_random]
                index = torch.cat((index, s_ind[0:classes_proportion[i].item()]), 0)
            index = index.type(torch.long)
            index = index.view(-1)
            index_rand = torch.randperm(len(index), generator=self.gen)
            index = index[index_rand]
            X_minibatch, y_minibatch = X[index], y[index].long()
        else:
            index = torch.randperm(len(X), generator=self.gen)
            index = index[:self.batchsize]
            X_minibatch = torch.tensor(X[index])
            y_minibatch = y
        return X_minibatch.float(), y_minibatch, index

    def g(self, x):
        return self.feature_extractor(x)

    def f(self, features_conv):
        h = features_conv.mean(dim=2, keepdim=False)  # Average pooling
        h = self.classifier(h)
        log_probas = self.logSoftmax(h)
        return log_probas

    def forward(self, x):
        features_conv = self.g(x)
        log_probas = self.f(features_conv)
        return log_probas, features_conv

    def train_iteration(self):
        self.train()
        self.new_iteration()
        self.optimizer.zero_grad()
        self.draw_batches()
        loss, _ = self.compute_total_loss()
        loss.backward()
        self.optimizer.step()

    def CE_similarity(self, labels_source, logSoftmax_target):
        """
        Cross-Entropy Similarity
        Compute a cross entropy between each pairs of label_source and logSoftmaxTarget
        :param labels_source: The true label of source batch
        :param logSoftmax_target: the prediction for target batch
        :return: a matrix (b x b') of costs
        """
        def to_onehot(y, n_classe=0):
            ncl = torch.max(torch.tensor([torch.max(y), n_classe-1]))
            n_values = ncl + 1
            return torch.eye(n_values)[y]
        logSoftmax_target = self.logSoftmax(logSoftmax_target)
        labels_source_onehot = to_onehot(labels_source, n_classe=self.n_classes)
        logSoftmax_target_trans = torch.transpose(logSoftmax_target, 1, 0)
        similarity_cross_entropy = -torch.matmul(labels_source_onehot, logSoftmax_target_trans)
        return similarity_cross_entropy

    def compute_total_loss(self, training=True):
        if training:
            logprobas_source, conv_features_source = self.forward(self.batch_source.transpose(1, 2))
            classif_loss = self.crossLoss(logprobas_source, self.batch_labels_source)
            self.loss_count.append(classif_loss.item())
            if self.save_latent_source:
                self.history_latent_source.append(conv_features_source.detach())
        else:
            logprobas_source, conv_features_source = self.forward(self.batch_source_valid.transpose(1, 2))
            classif_loss = self.crossLoss(logprobas_source, self.batch_labels_source_valid)
        return classif_loss, logprobas_source

    def fit_several_iterations(self):
        while self.iteration < self.max_iterations:
            t0 = time.time()
            self.train_iteration()
            t1 = time.time()
            self.training_time_.append(t1 - t0)
            if self.iteration % 2 == 0:
                if self.X_source_valid is not None:
                    valid_loss = self.unsupervised_validation_step(verbose_step=self.validation_step)
                    self.loss_count_valid.append(valid_loss.item())
                self.save_model()
            if self.iteration % self.validation_step == 0:
                print(self.iteration)
                if self.saving:
                    self.save_stuff()
        if self.saving:
            self.save_stuff()
        self.save_model()

    def fit(self, X_source, y_source, X_source_valid=None, y_source_valid=None, X_target_valid=None):

        self.X_source_ = X_source
        self.y_source_ = y_source
        self.n_classes = len(torch.unique(self.y_source_))

        self.X_source_valid = X_source_valid
        self.y_source_valid = y_source_valid
        self.X_target_valid = X_target_valid

        _, count_classes = torch.unique(self.y_source_, return_counts=True)
        sample_vec = torch.zeros(size=(self.n_classes,))
        for cl in range(0, self.n_classes):
            cl_bs = torch.round(self.batchsize * count_classes[cl] / torch.sum(count_classes))
            if cl_bs <= 1:
                cl_bs += 2
            sample_vec[cl] = cl_bs
        while sample_vec.sum() > self.batchsize:
            sample_vec[torch.argmax(sample_vec)] -= 1
        while sample_vec.sum() < self.batchsize:
            sample_vec[torch.argmin(sample_vec)] += 1
        self.sample_vec = sample_vec.type(torch.int)

        if self.y_source_valid is not None:
            batchsize_valid = torch.min(torch.Tensor([self.batchsize, len(self.y_source_valid)]))
            _, count_classes = torch.unique(self.y_source_valid, return_counts=True)
            sample_vec_valid = torch.zeros(size=(self.n_classes,))
            for cl in range(0, self.n_classes):
                cl_bs = torch.round(batchsize_valid * count_classes[cl] / torch.sum(count_classes))
                if cl_bs <= 1:
                    cl_bs += 2
                sample_vec_valid[cl] = cl_bs
            while sample_vec_valid.sum() > batchsize_valid:
                sample_vec_valid[torch.argmax(sample_vec_valid)] -= 1
            while sample_vec_valid.sum() < batchsize_valid:
                sample_vec_valid[torch.argmin(sample_vec_valid)] += 1
            self.sample_vec_valid = sample_vec_valid.type(torch.int)

        else:
            self.sample_vec_valid = None

        if self.CUDA_train:
            if torch.cuda.is_available():
                self.X_source_ = self.X_source_.cuda()
                self.y_source_ = self.y_source_.cuda()
                self.X_source_valid = self.X_source_valid.cuda()
                self.y_source_valid = self.y_source_valid.cuda()
                self.X_target_valid = self.X_target_valid.cuda()

        if self.max_iterations > 0:
            self.fit_several_iterations()

    def new_iteration(self):
        self.iteration += 1

    def unsupervised_validation_step(self, verbose_step=1):

        self.eval()
        self.draw_batches(training=False)
        loss, logprobas_source = self.compute_total_loss(training=False)
        pred = logprobas_source.data.max(1, keepdim=True)[1]

        correct = pred.eq(self.batch_labels_source_valid.data.view_as(pred)).cpu().sum()
        len_data = len(self.batch_labels_source_valid)
        self.acc_source.append(100. * correct / len_data)
        if self.iteration % verbose_step == 0:
            print(self.iteration, "Validation set :")
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss.detach().item(), correct, len_data,
                                                                           100. * correct / len_data))
        return loss

    def evaluate(self, inputs, labels, domain="target"):

        with torch.no_grad():
            self.eval()
            inputs = torch.tensor(inputs).type(torch.float)
            labels = torch.tensor(labels)
            if self.CUDA_train:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

            out, out_cnn = self.forward(inputs.transpose(1, 2))
            out_cnn_mean = out_cnn.mean(2)
            loss = self.crossLoss(out.float(), labels)
            pred = out.data.max(1, keepdim=True)[1]
            correct = pred.eq(labels.data.view_as(pred)).cpu().sum()
            if self.saving:
                names = [domain + "_rout_conv.npy", domain + "_out_conv.npy", domain + "_prediction.npy",
                         domain + "_target.npy", domain + "_confusion_mat.npy"]
                files = [out_cnn.cpu(), out_cnn_mean.cpu(), pred.cpu(), labels.cpu(),
                         sklearn.metrics.confusion_matrix(labels.cpu(), pred.cpu())]
                self.save_stuff(files=files, names=names)
            loss /= len(labels)

            self.acc_target.append(100. * correct / len(labels))
        print(self.name)
        print(self.iteration, "Evaluation set ", domain, ":")
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(loss, correct, len(labels),
                                                                       100. * correct / len(labels)))
        print("F1 micro score is : ", sklearn.metrics.f1_score(labels.cpu(), pred.cpu(), average="micro"))
        print("F1 macro score is : ", sklearn.metrics.f1_score(labels.cpu(), pred.cpu(), average="macro"))
        print("F1 weigthed score is : ", sklearn.metrics.f1_score(labels.cpu(), pred.cpu(), average="weighted"))
        return 100. * correct / len(labels)

    def predict_latent(self, X):
        with torch.no_grad():
            self.eval()
            inputs_source = X.float()
            sample_vec = torch.zeros(size=(self.n_classes,))
            inputs_source, labels_source, index_source = self.mini_batch_class_balanced(X=inputs_source,
                                                                                        classes_proportion=sample_vec,
                                                                                        y=labels_source)
            out_source, out_conv_source = self.forward(inputs_source.transpose(1, 2))
            names = ["Conv_source.npy"]
            files = [out_conv_source.cpu().numpy()]
            self.save_stuff(names=names, files=files, path=path)
            return out_conv_source

    def predict(self, X_target_test):
        with torch.no_grad():
            self.eval()
            inputs = torch.tensor(X_target_test).type(torch.float)

            out, out_cnn = self.forward(inputs.transpose(1, 2))
            prediction = out.data.max(1, keepdim=True)[1]
            names = ["target_prediction.npy"]
            files = [prediction.cpu()]
            self.save_stuff(files=files, names=names)
        return prediction

    def save_stuff(self, files=None, names=None, path=None):
        if path is None:
            path = self.name
        if files is None:
            files = self.saving_files
        if names is None:
            names = self.saving_names
        for stuff in range(0, len(files)):
            if names[stuff]=="history_DTW_matrices.npy":
                torch.save(files[stuff], path + str(self.iteration) + "history_DTW_matrices.pt")
            else:    
                np.save(path + str(self.iteration) + names[stuff], files[stuff])

    def save_model(self):
        torch.save(self.state_dict(), self.name + str(self.iteration) + '.pt')

    @staticmethod
    def torch2numpy(list_):
        list_return = []
        for stuff in list_:
            list_return.append(stuff.cpu().numpy())
        return list_return

    def get_params(self, deep=True):
        return super().get_params(deep)

    def set_params(self, **params):
        return super().set_params(**params)

    def score(self, X, y, sample_weight=None):
        with torch.no_grad():
            self.eval()
            inputs = torch.tensor(X).type(torch.float)
            labels = torch.tensor(y)

            out, out_cnn = self.forward(inputs.transpose(1, 2))
            loss = self.crossLoss(out.float(), labels)
            pred = out.data.max(1, keepdim=True)[1]
            correct = pred.eq(labels.data.view_as(pred)).cpu().sum()
            loss /= len(labels)
        return 100. * correct / len(labels)


def calcul_torch_cdist(batch_source, batch_target):
    """
    Computes the global cost l2² for the DTWs in torch so that we can keep the gradient for backward
    :param batch_source: a batch of size (b x T x q)
    :param batch_target: a batch of size (b' x T' x q)
    """
    batch_source_flat = torch.reshape(batch_source, (batch_source.shape[0] * batch_source.shape[1], batch_source.shape[-1]))
    batch_target_flat = torch.reshape(batch_target, (batch_target.shape[0] * batch_target.shape[1], batch_target.shape[-1]))
    dtw_cost_flat = torch.cdist(batch_source_flat, batch_target_flat)
    dtw_cost = torch.reshape(dtw_cost_flat, (batch_source.shape[0], batch_target.shape[0], batch_source.shape[1], batch_target.shape[1]))
    dtw_cost = dtw_cost.transpose(1, 0)
    return dtw_cost


@jit
def dtw(x, y, dtw_cost):
    """
    Computes the DTW between time series x and y using pre-computed cost.
    :param x: a time series of size (T, x q)
    :param y: a time series of size (T' x q)
    :param dtw_cost: the corresponding l2² cost for the pair of series x and y
    :return: the cumulative sum for the dtw path
    """
    l1 = x.shape[0]
    l2 = y.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = dtw_cost[i, j]
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])
    return cum_sum[1:, 1:]


@jit
def _return_path(acc_cost_mat):
    """
    From the cumulative sum obtain with dtw, yields a matrix that contains the dtw path
    """
    sz1, sz2 = acc_cost_mat.shape
    matrix_path = np.zeros(shape=(sz1, sz2))
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
            matrix_path[0, j-1] = 1
        elif j == 0:
            path.append((i - 1, 0))
            matrix_path[i-1, 0] = 1
        else:
            arr = np.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
                matrix_path[i-1, j-1] = 1
            elif argmin == 1:
                path.append((i - 1, j))
                matrix_path[i-1, j] = 1
            else:
                path.append((i, j - 1))
                matrix_path[i, j-1] = 1
    return matrix_path


@jit(nopython=True, parallel=True)
def DTW_batch(batch_source, batch_target, dtw_cost):
    """
    for each pair of series in batches source and target, computes the corresponding DTW using the dtw_cost
    :param batch_source: a batch of size (b x T x q)
    :param batch_target: a batch of size (b' x T' x q)
    :param dtw_cost: the l2² distances between each pair of series
    :return: all the dtw path matrices
    """
    all_path = np.empty(shape=(batch_source.shape[0], batch_target.shape[0], batch_source.shape[1], batch_target.shape[1]))
    for s in prange(0, batch_source.shape[0]):
        for t in prange(0, batch_target.shape[0]):
            dtw_matrix = dtw(batch_source[s], batch_target[t], dtw_cost[s, t])
            path_matrix = _return_path(dtw_matrix)
            all_path[s, t] = path_matrix
    return all_path


def Cost_matrix_torch_quick(all_path, all_cost):
    """
    Computes the cost that will be used in the loss function of the main training
    :param all_path: all the paths of the current batches
    :param all_cost: the associated costs
    :return: a matrix of size (batchsize source x batchsize target) containing the dtw costs
    """
    dtw_cost = torch.sum(all_path * all_cost, dim=(2, 3))
    return dtw_cost


def OT(cost, weight_X, weight_Y):
    """
    Compute the optimal transport plan (earth mover distance)
    :param cost: the pre-computed dtw costs
    :param weight_X: distribution of X mass
    :param weight_Y: distribution of y mass
    :return: the transport plan matrix of size (batchsize source x batchsize target)
    """
    with torch.no_grad():
        gamma = emd(weight_X, weight_Y, cost)
    return gamma


class DeepJDOT_loss(nn.Module):
    def __init__(self, alpha, beta, batchsize, target_prop=None):
        """
        The DeepJDOT_MAD loss function.
        It computes all dtw path between batches source and target of time series and uses these dtw scores to compute
        an optimal transport matrix between the batches that will be used to optimize the main neural network.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.batchsize = batchsize
        self.target_prop = target_prop

    def forward(self, out_conv_source, out_conv_target, labels_source, similarity_CE):
        dtw_cost_torch = calcul_torch_cdist(out_conv_source, out_conv_target)
        dtw_cost_numpy = dtw_cost_torch.detach().numpy()
        all_path = DTW_batch(out_conv_source.detach().cpu().numpy(), out_conv_target.detach().cpu().numpy(),
                             dtw_cost_numpy)
        all_path_torch = torch.from_numpy(all_path).type(torch.float32)
        dtw_cost = Cost_matrix_torch_quick(all_path_torch, dtw_cost_torch)
        cost_OT = self.alpha * dtw_cost + self.beta * similarity_CE
        weight_Y = torch.ones(out_conv_target.shape[0]) / out_conv_target.shape[0]
        weight_X = torch.ones(out_conv_source.shape[0]) / out_conv_source.shape[0]
        if self.target_prop is not None:
            for cl in range(0, len(torch.unique(labels_source))):
                weight_X[labels_source == cl] = self.target_prop[cl] / len(labels_source[labels_source == cl])

        gamma = OT(cost_OT, weight_X, weight_Y)
        alpha_cost = (gamma * dtw_cost).sum()
        beta_cost = (gamma * similarity_CE).sum()
        length = (out_conv_source.shape[-1] + out_conv_target.shape[-1]) / 2
        return self.alpha * alpha_cost / length, self.beta * beta_cost / length, gamma


class CNN_DeepJDOT(Basic_CNN):
    def __init__(self, 

                 batchsize, 
                 feature_extractor, 
                 classifier,
                 name="Default_name", 
                 X_target=None,
                 
                 y_target=None,
                 lr=0.001, 
                 saving=True,
                 max_iterations=3000,
                 validation_step=1000,
    
                 target_prop=None,
                 alpha=0.01, 
                 beta=0.01,
                 save_latent_source=False,
                 save_latent_target=False,

                 save_OT_plan=True
                 ):

        """
        :param name: path to save the files
        :param batchsize:
        :param feature_extractor: the feature extractor as a nn.sequential
        :param classifier: the classifier as a nn.sequential
        :param alpha: the weight of MAD in the loss function
        :param beta: the weight of the cross similarity label/prediction in the loss function
        :param lr: Learning rate
        :param saving: If it is important to save the model or not
        :param target_prop: Do we know the proportion of the target labels ?
        In case we are doing weakly supervised learning
        """
        super().__init__(name=name, 
                         batchsize=batchsize,
                         feature_extractor=feature_extractor, 
                         classifier=classifier, 
                         X_target=X_target,
                         y_target=y_target,
                         lr=lr, 
                         saving=saving, 
                         max_iterations=max_iterations, 
                         validation_step=validation_step, 
                         CUDA_train=False,
                         save_latent_source=save_latent_source)

        self.alpha = alpha
        self.beta = beta
        self.target_prop = target_prop
        self.general_loss = DeepJDOT_loss(alpha=self.alpha, beta=self.beta, batchsize=self.batchsize,
                                  target_prop=self.target_prop)
        
        self.loss_beta = []
        self.saving_names.append("loss_beta.npy")
        self.saving_files.append(self.loss_beta)

        self.loss_alpha = []
        self.saving_names.append("loss_alpha.npy")
        self.saving_files.append(self.loss_alpha)

        self.acc_target = []
        self.saving_names.append("acc_target.npy")
        self.saving_files.append(self.acc_target)

        self.save_latent_target = save_latent_target
        if self.save_latent_target:
            self.history_latent_target = []
            self.saving_files.append(self.history_latent_target)
            self.saving_names.append("history_latent_target.npy")
        
        self.save_OT_plan=save_OT_plan
        if self.save_OT_plan:
            self.history_OT_plan = []
            self.saving_files.append(self.history_OT_plan)
            self.saving_names.append("history_OT_plan.npy")

    def draw_batches(self, training=True):
        if training:
            self.batch_source, self.batch_labels_source, self.batch_index_source = self.mini_batch_class_balanced(X=self.X_source_,
                                                                                                                classes_proportion=self.sample_vec,
                                                                                                                y=self.y_source_)
            self.batch_target, self.batch_labels_target, self.batch_index_target = self.mini_batch_class_balanced(X=self.X_target,
                                                                                                                classes_proportion=self.sample_vec)
        else:
            self.batch_source_valid, self.batch_labels_source_valid, self.batch_index_source_valid = self.mini_batch_class_balanced(X=self.X_source_valid,
                                                                                                                                    classes_proportion=self.sample_vec_valid,
                                                                                                                                    y=self.y_source_valid)
            self.batch_target_valid, self.batch_labels_target_valid, self.batch_index_target_valid = self.mini_batch_class_balanced(X=self.X_target_valid,
                                                                                                                                    classes_proportion=self.sample_vec_valid)

    def compute_total_loss(self, training=True):
        """
        Compute the 3 elements of the MAD loss
        :param X_source:
        :param y_source:
        :param X_target:
        :param training:
        :return:
        """
        if training:
            logprobas_source, conv_features_source = self.forward(self.batch_source.transpose(1, 2))
            classif_loss = self.crossLoss(logprobas_source, self.batch_labels_source)
            loss = classif_loss
            self.loss_count.append(classif_loss.item())
            if (self.alpha != 0) or (self.beta != 0):
                logprobas_target, conv_features_target = self.forward(self.batch_target.transpose(1, 2))
                similarity_CE = self.CE_similarity(labels_source=self.batch_labels_source, logSoftmax_target=logprobas_target)
                alpha_loss, beta_loss, self.OT_ = self.general_loss(conv_features_source.transpose(1, 2), 
                                                                    conv_features_target.transpose(1, 2), 
                                                                    self.batch_labels_source, 
                                                                    similarity_CE)
                self.loss_alpha.append(alpha_loss.item())
                self.loss_beta.append(beta_loss.item())
                if self.save_OT_plan:
                    self.history_OT_plan.append(self.OT_)
                if self.save_latent_source:
                    self.history_latent_source.append(conv_features_source.detach())
                if self.save_latent_target:
                    self.history_latent_target.append(conv_features_target.detach())
                loss += alpha_loss + beta_loss
        else:
            logprobas_source, conv_features_source = self.forward(self.batch_source_valid.transpose(1, 2))
            classif_loss = self.crossLoss(logprobas_source, self.batch_labels_source_valid)
            loss = classif_loss
            self.loss_count.append(classif_loss.item())
            if (self.alpha != 0) or (self.beta != 0):
                logprobas_target, conv_features_target = self.forward(self.batch_target_valid.transpose(1, 2))
                similarity_CE = self.CE_similarity(labels_source=self.batch_labels_source_valid, logSoftmax_target=logprobas_target)
                alpha_loss, beta_loss, self.OT_ = self.general_loss(conv_features_source.transpose(1, 2), 
                                                                    conv_features_target.transpose(1, 2), 
                                                                    self.batch_labels_source_valid, 
                                                                    similarity_CE)
                loss += alpha_loss + beta_loss
        return loss, logprobas_source


class CNNMAD(Basic_CNN):
    def __init__(self, 

                 batchsize,
                 feature_extractor, 
                 classifier,
                 name="Default_name",              
                 X_target=None,

                 y_target=None,
                 lr=0.001, 
                 saving=True,
                 max_iterations=3000, 
                 validation_step=1000,
                 
                 target_prop=None,
                 alpha=0.01, 
                 beta=0.01,
                 MAD_class=True,
                 save_OT_plan=False,
                 
                 save_DTW_matrices=True,
                 save_latent_source=False,
                 save_latent_target=False
                 ):

        """
        See section "Neural domain adaptation with a MAD loss" in the paper for more details

        :param name: path to save the files
        :param batchsize:
        :param feature_extractor: the feature extractor as a nn.sequential
        :param classifier: the classifier as a nn.sequential
        :param alpha: the weight of MAD in the loss function
        :param beta: the weight of the cross similarity label/prediction in the loss function
        :param lr: Learning rate
        :param saving: If it is important to save the model or not
        :param target_prop: Do we know the proportion of the target labels ?
        In case we are doing weakly supervised learning
        """
        super().__init__(name=name, 
                         batchsize=batchsize,
                         feature_extractor=feature_extractor, 
                         classifier=classifier,
                         X_target=X_target,
                         y_target=y_target,
                         lr=lr,
                         saving=saving, 
                         max_iterations=max_iterations, 
                         validation_step=validation_step, 
                         CUDA_train=False,
                         save_latent_source=save_latent_source)

        self.X_target = X_target
        self.alpha = alpha
        self.beta = beta
        self.MAD_class = MAD_class
        self.target_prop = target_prop

        self.general_loss = MAD_loss(MAD_class=MAD_class, 
                                     alpha=self.alpha, 
                                     beta=self.beta, 
                                     target_prop=self.target_prop)

        self.loss_beta = []
        self.saving_names.append("loss_beta.npy")
        self.saving_files.append(self.loss_beta)

        self.loss_alpha = []
        self.saving_names.append("loss_alpha.npy")
        self.saving_files.append(self.loss_alpha)

        self.acc_target = []
        self.saving_names.append("acc_target.npy")
        self.saving_files.append(self.acc_target)

        self.save_latent_target = save_latent_target
        if self.save_latent_target:
            self.history_latent_target = []
            self.saving_files.append(self.history_latent_target)
            self.saving_names.append("history_latent_target.npy")
        
        self.save_OT_plan=save_OT_plan
        if self.save_OT_plan:
            self.history_OT_plan = []
            self.saving_files.append(self.history_OT_plan)
            self.saving_names.append("history_OT_plan.npy")

        self.save_DTW_matrices=save_DTW_matrices
        if self.save_DTW_matrices:
            self.history_DTW_matrices = []
            self.saving_files.append(self.history_DTW_matrices)
            self.saving_names.append("history_DTW_matrices.npy")

    def draw_batches(self, training=True):
        if training:
            self.batch_source, self.batch_labels_source, self.batch_index_source = self.mini_batch_class_balanced(X=self.X_source_,
                                                                                                                classes_proportion=self.sample_vec,
                                                                                                                y=self.y_source_)
            self.batch_target, self.batch_labels_target, self.batch_index_target = self.mini_batch_class_balanced(X=self.X_target,
                                                                                                                classes_proportion=self.sample_vec)
        else:
            self.batch_source_valid, self.batch_labels_source_valid, self.batch_index_source_valid = self.mini_batch_class_balanced(X=self.X_source_valid,
                                                                                                                                    classes_proportion=self.sample_vec_valid,
                                                                                                                                    y=self.y_source_valid)
            self.batch_target_valid, self.batch_labels_target_valid, self.batch_index_target_valid = self.mini_batch_class_balanced(X=self.X_target_valid,
                                                                                                                                    classes_proportion=self.sample_vec_valid)
    
    def compute_total_loss(self, training=True):
        """Compute the 3 elements of the MAD loss
        See equation 8 in the paper

        :param X_source:
        :param y_source:
        :param X_target:
        :param training:
        :return:
        """
        if training:
            logprobas_source, conv_features_source = self.forward(self.batch_source.transpose(1, 2))
            classif_loss = self.crossLoss(logprobas_source, self.batch_labels_source)
            loss = classif_loss
            self.loss_count.append(classif_loss.item())
            sample_classes_proportion = self.sample_vec
            if (self.alpha != 0) or (self.beta != 0):
                logprobas_target, conv_features_target = self.forward(self.batch_target.transpose(1, 2))
                similarity_CE = self.CE_similarity(labels_source=self.batch_labels_source, logSoftmax_target=logprobas_target)
                alpha_loss, beta_loss, self.OT_, self.DTW_ = self.general_loss(conv_features_source, 
                                                                            conv_features_target, 
                                                                            self.batch_labels_source, 
                                                                            similarity_CE, 
                                                                            sample_classes_proportion)
                loss += alpha_loss + beta_loss
                self.loss_alpha.append(alpha_loss.item())
                self.loss_beta.append(beta_loss.item())
                if self.save_OT_plan:
                    self.history_OT_plan.append(self.OT_)
                if self.save_DTW_matrices:
                    self.history_DTW_matrices.append(self.DTW_)
                if self.save_latent_source:
                    self.history_latent_source.append(conv_features_source.detach())
                if self.save_latent_target:
                    self.history_latent_target.append(conv_features_target.detach())
        else:
            logprobas_source, conv_features_source = self.forward(self.batch_source_valid.transpose(1, 2))
            classif_loss = self.crossLoss(logprobas_source, self.batch_labels_source_valid)
            loss = classif_loss
            sample_classes_proportion = self.sample_vec_valid
            if (self.alpha != 0) or (self.beta != 0):
                logprobas_target, conv_features_target = self.forward(self.batch_target_valid.transpose(1, 2))
                similarity_CE = self.CE_similarity(labels_source=self.batch_labels_source_valid, logSoftmax_target=logprobas_target)
                alpha_loss, beta_loss, self.OT_, self.DTW_ = self.general_loss(conv_features_source, 
                                                                            conv_features_target, 
                                                                            self.batch_labels_source_valid, 
                                                                            similarity_CE, 
                                                                            sample_classes_proportion)
                loss += alpha_loss + beta_loss
        return loss, logprobas_source

    def forward_MAD(self, X_source, y_source, X_target, y_target=None, train_test="test", path=None):
        """
        Passes the data through the feature extractor to obtain the MAD transport matrix and the DTW matrices.
        """
        with torch.no_grad():
            self.eval()
            inputs_source = X_source.float()
            labels_source = y_source
            inputs_target = X_target.float()
            labels_target = y_target
            _, count_classes = torch.unique(labels_source, return_counts=True)
            sample_vec = torch.zeros(size=(self.n_classes,))
            for cl in range(0, self.n_classes):
                cl_bs = torch.round(self.batchsize * count_classes[cl] / torch.sum(count_classes))
                if cl_bs <= 1:
                    cl_bs += 1
                sample_vec[cl] = cl_bs
            while sample_vec.sum() > self.batchsize:
                sample_vec[torch.argmax(sample_vec)] -= 1
            sample_vec = sample_vec.type(torch.int)
            inputs_source, labels_source, index_source = self.mini_batch_class_balanced(X=inputs_source,
                                                                                        classes_proportion=sample_vec,
                                                                                        y=labels_source)
            inputs_target, labels_target, index_target = self.mini_batch_class_balanced(X=inputs_target,
                                                                            classes_proportion=sample_vec, y=labels_target)
            out_target, out_conv_target = self.forward(inputs_target.transpose(1, 2))
            out_source, out_conv_source = self.forward(inputs_source.transpose(1, 2))
            similarity_CE = self.CE_similarity(labels_source=labels_source, logSoftmax_target=out_target)
            self.general_loss.forward(out_conv_source, out_conv_target, labels_source, similarity_CE, sample_vec)
            pred = out_target.data.max(1, keepdim=True)[1]
            names = [train_test + 'DTW_forward_MAD.npy', 
                     train_test + 'OT_forward_MAD.npy',
                     train_test + 'OT_Cost_forward_MAD.npy', 
                     train_test + 'Conv_target.npy',
                     train_test + "Conv_source.npy", 
                     train_test + 'labels_target.npy', 
                     train_test + 'labels_source.npy',
                     train_test + "pred_forward.npy", 
                     train_test + "batch_source.npy", 
                     train_test + "batch_target.npy"]

            files = [self.torch2numpy(self.general_loss.DTW_), 
                     self.general_loss.OT_.cpu().numpy(),
                     self.general_loss.cost_OT_.cpu().numpy(),
                     out_conv_target.cpu().numpy(), 
                     out_conv_source.cpu().numpy(), 
                     labels_target.cpu().numpy(),
                     labels_source.cpu().numpy(),
                     pred.cpu().numpy(), 
                     inputs_source.cpu().numpy(), 
                     inputs_target.cpu().numpy()]
            self.save_stuff(names=names, files=files, path=path)

