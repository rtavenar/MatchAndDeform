from General_model import Basic_CNN, CNN_DeepJDOT, CNNMAD
import argparse
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import os
import warnings
from dataset_loader import load_a_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_id', type=int, help='The source domain')
    parser.add_argument('-t', '--target_id', type=int, help='The target domain')
    parser.add_argument('-data', '--dataset_name', type=str, help="Which dataset to take")
    parser.add_argument("-a", "--alpha", type=float, help="Alpha")
    parser.add_argument("-b", "--beta", type=float, help='Beta')
    parser.add_argument('-lr', "--learning_rate", type=float, help="The learning rate")
    parser.add_argument('-i', "--iteration", type=int)
    parser.add_argument('-n', "--name_model", type=str)
    parser.add_argument('-dtw', "--per_class_dtw", type=str)
    parser.add_argument("-sd", "--seed", type=int)
    parser.add_argument("-tp", "--target_class_proportion", type=str)
    parser.add_argument('-ba', "--batchsize", type=int)
    parser.add_argument('-m', "--model", type=str)

    bn_affine = False #This has empirically been set to false.
    args, _ = parser.parse_known_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        npr.seed(args.seed)

    DTW_multiple = True
    if args.per_class_dtw == "False":
        DTW_multiple = False

    if args.batchsize is None:
        batchs = 256
    else:
        batchs = args.batchsize

    def to_onehot(y):
        n_values = np.max(y) + 1
        return np.eye(n_values)[y]


    def from_numpy_to_torch(filename, float_or_long=True):
        data = np.load(filename)
        data_t = torch.from_numpy(data)
        if float_or_long:
            data_t = data_t.type(torch.float)
        else:
            data_t = data_t.type(torch.long)
        return data_t


    source = str(args.source_id)
    target = str(args.target_id)

    if args.dataset_name == 'HHAR':
        chan = 3
        n_classes = 6
        dataset_path = "Dataset/HHAR/"
    if args.dataset_name == 'ucihar':
        chan = 9
        n_classes = 6
        dataset_path = 'Dataset/HAR/'
    if args.dataset_name == "UWAVE":
        n_classes = 8
        chan = 3
        dataset_path = 'Dataset/Uwave/'
    if args.dataset_name == "TarnBrittany":
        n_classes = 5
        chan = 10
        dataset_path = 'Dataset/TarnBrittany/'
    if args.dataset_name == "TinyTimeMatch":
        n_classes = 7
        chan = 10
        dataset_path = 'Dataset/TinyTimeMatch/'
    if args.dataset_name == "miniTimeMatch":
        n_classes = 8
        chan = 10
        dataset_path = 'Dataset/miniTimeMatch/'

    
    train_source, train_source_label, valid_source, valid_source_label, test_source, test_source_label = load_a_dataset(dataset_name=dataset_path + args.dataset_name, 
                                                                                                                        domain_id=source)
    train_target, train_target_label, valid_target, valid_target_label, test_target, test_target_label = load_a_dataset(dataset_name=dataset_path + args.dataset_name, 
                                                                                                                        domain_id=target)

    path_save = os.path.join(args.dataset_name, args.model, str(args.alpha), str(args.beta), str(args.learning_rate),
                             str(args.source_id) + "_" + str(args.target_id))

    if os.path.exists(path_save) is False:
        os.makedirs(path_save)
    name = path_save + "/" + args.name_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=chan, out_channels=128, kernel_size=8, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(num_features=128, affine=bn_affine),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(num_features=256, affine=bn_affine),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm1d(num_features=128, affine=bn_affine),
            nn.ReLU())
        classifier = nn.Sequential(nn.Linear(128, n_classes))  # /!\ Does not include softmax activation

        _, c = np.unique(train_target_label, return_counts=True)

        target_prop = None
        if args.target_class_proportion == "True":
            target_prop = c / train_target.shape[0]
            target_prop = torch.Tensor(target_prop)
        source_size = train_source.shape[0]
        target_size = train_target.shape[0]
        batchsize = torch.min(torch.tensor([batchs, source_size, target_size]))
        batchsize = batchsize.item()

        if args.model == "DeepJDOT_DTW":
            CNN_mod = CNN_DeepJDOT(name=name, 
                                   batchsize=batchsize,
                                   feature_extractor=feature_extractor,
                                   classifier=classifier, 
                                   X_target=train_target,

                                   y_target=train_target_label,
                                   lr=args.learning_rate, 
                                   saving=True, 
                                   max_iterations=args.iteration,
                                   validation_step=1000,

                                   target_prop=target_prop,
                                   alpha=args.alpha,
                                   beta=args.beta,
                                   save_latent_source=True,
                                   save_latent_target=True,

                                   save_OT_plan=True,
                                   )
            #Allocation of CPUs for training DeepJDOT_DTW or MAD (no GPU is used during this training)
            cpu_num = 8
            torch.set_num_threads(cpu_num)
            torch.cuda.is_available = lambda: False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.model == "MAD":
            CNN_mod = CNNMAD(name=name,
                             batchsize=batchsize, 
                             feature_extractor=feature_extractor,
                             classifier=classifier, 
                             X_target=train_target,

                             y_target=train_target_label, 
                             lr=args.learning_rate, 
                             saving=True, 
                             max_iterations=args.iteration, 
                             validation_step=2,
                             
                             target_prop=target_prop,
                             alpha=args.alpha,
                             beta=args.beta,
                             MAD_class=DTW_multiple, 
                             save_OT_plan=False,

                             save_DTW_matrices=False,
                             save_latent_source=False,
                             save_latent_target=False)

            #Allocation of CPUs for training DeepJDOT_DTW or MAD (no GPU is used during this training)
            cpu_num = 8
            torch.set_num_threads(cpu_num)
            torch.cuda.is_available = lambda: False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.model == "Basic":
            CNN_mod = Basic_CNN(name=name,
                                batchsize=batchsize, 
                                feature_extractor=feature_extractor, 
                                classifier=classifier, 
                                X_target=train_target,

                                y_target=train_target_label,
                                lr=args.learning_rate, 
                                saving=True,
                                max_iterations=args.iteration,
                                validation_step=1000,
                                
                                CUDA_train=True,
                                save_latent_source=True
                                )
        CNN_mod.fit(X_source=train_source, 
                    y_source=train_source_label, 
                    X_source_valid=valid_source, 
                    y_source_valid=valid_source_label, 
                    X_target_valid=valid_target)

        CNN_mod.evaluate(inputs=test_target, 
                         labels=test_target_label)
