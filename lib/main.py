import argparse
from utils import load_data, load_data_SMOTE
import cnn

def main(args):
    if args.experiment == 3:
        print('DATA oversampling with SMOTE')
        train_dataset = load_data_SMOTE('cnn', split='train')
        test_dataset = load_data_SMOTE('cnn', split='test')
    else:
        train_dataset = load_data('cnn', split='train')
        test_dataset = load_data('cnn', split='test')
    cnn.fit(experiment=args.experiment,
             train_dataset=train_dataset,
              test_dataset=test_dataset)
    
        
if __name__ == '__main__':
    '''
    Comparaison study with MNIST
    Experiment 0: No oversampling
    Experiment 1: Oversample minor class to be balanced
    Experiment 2: Use a cost-sensitive loss function
    Experiment 3: SMOTE
    Experiment 4: Valid oversampling

    Model: CNN
    Evaluataion measures: sensitivity, specificity, accuracy, AA

    This code assumes that minor class is encoded as 1, so that P(Y=1) < 0.5.
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--experiment', type=int, default=1,
                          choices = [0,1,2,3,4], help='experiment to be conducted')
    args = parser.parse_args()

    print('-'*10)
    print('Hparams: ', args)
    main(args)


    