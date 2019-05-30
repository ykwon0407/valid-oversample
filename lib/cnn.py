import torch
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
from sampler import ImbalancedDatasetSampler
from networks import Simple_Net
from utils import evaluate_prediction
import os
from valid_oversample import VO_LPD

hp_cnn = dict(
    batch_size=100,
    test_batch_size=1000,
    epochs=20,
    lr=0.005,
    momentum=0.5,
    no_cuda=False,
    seed=1004,
    log_interval=2,
    save_model=True,
    n_workers=12,
    gpu_device=1,
)

def train(args, model, device, train_loader, optimizer, epoch, loss_fct):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = loss_fct(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, loss_fct_te = torch.nn.NLLLoss(reduction='sum'), verbose=False):
    model.eval()
    test_loss, correct = 0, 0
    pred_list,true_list=[], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += loss_fct_te(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            true_list.append(target.cpu().numpy().ravel())
            pred_list.append(pred.cpu().numpy().ravel())

    test_loss /= len(test_loader.dataset)
    true_list, pred_list = np.hstack(true_list), np.hstack(pred_list)
    sensitivity, specificity, accuracy, average_accuracy = evaluate_prediction(true_list, pred_list)

    if verbose is True:
        print('Test set: Sensitivity: {:.4f}, Specificity: {:.4f}, Accuracy: {:.4f}, AA: {:.4f}'.format(
            sensitivity, specificity, accuracy, average_accuracy))


def extract_feature(model, device, train_loader, test_loader):

    if not os.path.exists('../results/feature_train.npy'):
        model.eval()
        feature_list,Y_list=[], []
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                feature, _ = model(data)

                Y_list.append(target.cpu().numpy().ravel())
                feature_list.append(feature.cpu().numpy().reshape(-1,200))

        Y_list, feature_list = np.hstack(Y_list).reshape(-1,1), np.vstack(feature_list)
        print('train shape: ', Y_list.shape, feature_list.shape)

        arr = np.hstack([feature_list, Y_list])
        file = '../results/feature_train.npy'
        np.save(file=file, arr=arr)

    if not os.path.exists('../results/feature_test.npy'):
        model.eval()
        feature_list,Y_list=[], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                feature, _ = model(data)

                Y_list.append(target.cpu().numpy().ravel())
                feature_list.append(feature.cpu().numpy().reshape(-1,200))

        Y_list, feature_list = np.hstack(Y_list).reshape(-1,1), np.vstack(feature_list)
        print('test shape: ', Y_list.shape, feature_list.shape)

        arr = np.hstack([feature_list, Y_list])
        file = '../results/feature_test.npy'
        np.save(file=file, arr=arr)

    data_train=np.load('../results/feature_train.npy')
    data_test=np.load('../results/feature_test.npy')
    X_tr, Y_tr = data_train[:,:-1], data_train[:,-1]
    X_te, Y_te = data_test[:,:-1], data_test[:,-1]

    return X_tr, Y_tr, X_te, Y_te


def fit(experiment=1, train_dataset=None, test_dataset=None):
    '''
    This code trains CNN with MNIST dataset
    '''
    assert (train_dataset is not None) or (test_dataset is not None), 'Check datasets.'
    args = SimpleNamespace(**hp_cnn)
    print('Hparams:', args)

    '''
    CUDA related arguments
    '''
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_device) if use_cuda else "cpu")
    kwargs = {'num_workers': args.n_workers, 'pin_memory': True} if use_cuda else {}

    '''
    Build Neural network
    '''
    model = Simple_Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    '''
    Define train_loader and test_loader
    '''
    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    shuffle=False,
                    batch_size=args.test_batch_size,
                    **kwargs)

    if experiment == 0:
        print('Method: No oversampling')
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=args.batch_size,
                        **kwargs)
        loss_fct_tr = torch.nn.NLLLoss()
    elif experiment == 1:
        print('Method: Oversample minor class data with 0.5')
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        sampler=ImbalancedDatasetSampler(train_dataset),
                        batch_size=args.batch_size,
                        **kwargs)
        loss_fct_tr = torch.nn.NLLLoss()
    elif experiment == 2:
        print('Method: Const-sensitive learning')
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=args.batch_size,
                        **kwargs)
        weights = torch.Tensor([1.0,9.0]).to(device)
        loss_fct_tr = torch.nn.NLLLoss(weight=weights)
    elif experiment == 3:
        print('Method: SMOTE method')
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=args.batch_size,
                        **kwargs)
        loss_fct_tr = torch.nn.NLLLoss()
    elif experiment == 4:
        print('Method: Valid oversampling')
        model_path="../results/mnist_cnn_ex-0.pt"
        if not os.path.exists(model_path):
            assert False, 'Save a model first. Conduct experiment 0'
        print('Load a model from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))

        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=False,
                        batch_size=args.batch_size,
                        **kwargs)
        X_tr, Y_tr, X_te, Y_te = extract_feature(model, device, train_loader, test_loader)

        '''
        1. calculate optimal target proportion p_sugg
        2. Oversample data
        3. calculate performance
        '''
        vo_lpd = VO_LPD(X_tr, Y_tr, spec_limit=0.03, sen_limit=0.05)
        p_sugg = vo_lpd.suggest_valid_prop()
        print('Proposed p_star is {}'.format(p_sugg))
        vo_lpd.oversample_data()
        vo_lpd.calculate_performance(X_te, Y_te)
        
    else:
        assert False, 'Check --experiment'

    '''
    Train a model
    '''
    if experiment != 4:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, loss_fct_tr)
            if epoch != args.epochs:
                test(model, device, test_loader)
            else:
                test(model, device, test_loader, verbose=True)

        save_path="../results/mnist_cnn_ex-{}.pt".format(experiment)
        if (args.save_model is True) and (os.path.exists(save_path) is not True):
            print('save model at {}'.format(save_path))
            torch.save(model.state_dict(),save_path)

    