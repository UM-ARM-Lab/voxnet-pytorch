import torch
import torch.nn as nn
#import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable
import torchvision.transforms as transforms

import imp
import logging
from path import Path
import numpy as np
import time
import os
import sys
import importlib
import argparse
import csv
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
from datasets.modelnet import ModelNet
from datasets.rope_dataset import RopeDataset

def main(args):
    # load network
    print("loading module")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    module = importlib.import_module("models."+args.model)
    #if args.rope_data:
    #    input_shape = (34, 34, 34)
    #else:
    input_shape = (32, 32, 32)

    if args.model == 'voxnet_multientry':
        model = module.VoxNetMultiEntry(num_classes=args.num_classes, input_shape=input_shape,
                                        use_same_net=args.use_same_net,
                                        num_grids=args.num_channels,
                                        device=device)
    else:
        model = module.VoxNet(num_classes=args.num_classes,
                              input_shape=input_shape,
                              num_channels=args.num_channels)
    model.to(device)

    # backup files
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    os.system('cp {} {}'.format(os.path.join(ROOT_DIR, 'models', args.model+'.py'), args.log_dir))
    os.system('cp {} {}'.format(__file__, args.log_dir))
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    #logging.info('logs will be saved to {}'.format(args.log_fname))
    #logger = Logger(args.log_fname, reinitialize=True)
    print("loading dataset")

    global LOG_FOUT
    LOG_FOUT = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    print(args)


    if args.rope_data:
        #data_dir = '/mnt/big_narstie_data/dmcconac'
        #data_dir +='/transition_learning_data_generation/smmap_generated_plans'
        #data_dir +='/rope_hooks_simple/generate_training_examples/raw_data'
        data_dir = '../../data'
        dset_train = RopeDataset(data_dir + '/train', samples_per_file=1024)
        dset_test = RopeDataset(data_dir + '/test', samples_per_file=1024)
        # Load labels separately to do weighted sampling
        with open(data_dir + '/training_labels.csv', 'r') as infile:
            csv_reader = csv.reader(infile, delimiter=',')
            labels = []
            for line in csv_reader:
                if len(line) == 2:
                    labels.append(int(line[1]))
        labels = np.asarray(labels)

    else:
        dset_train = ModelNet(os.path.join(ROOT_DIR, "data"),
                             args.training_fname,
                            duplicate_channels=args.num_channels)
        dset_test = ModelNet(os.path.join(ROOT_DIR, "data"),
                             args.testing_fname,
                            duplicate_channels=args.num_channels)

    fraction_positive = float(np.sum(labels)) / len(labels)
    print(fraction_positive)
    weights = np.ones(len(labels))
    weights[labels==1] *= 1 - fraction_positive
    weights[labels==0] *= fraction_positive
    print(weights)
    sampler = WeightedRandomSampler(weights=weights,
                                    num_samples=len(weights),
                                    replacement=True)

    train_loader = DataLoader(dset_train, batch_size=args.batch_size,
                              shuffle=False, sampler=sampler, drop_last=True,
                              num_workers=32)
    test_loader = DataLoader(dset_test, batch_size=args.batch_size,
                             num_workers=32)

    start_epoch = 0
    best_acc = 0.
    if args.cont:
        start_epoch, best_acc = load_checkpoint(args, model)

    print("set optimizer")
    # set optimization methods
    if args.num_classes == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, args.decay_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(start_epoch, args.max_epoch):
        scheduler.step()
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, args.max_epoch))
        start = time.time()

        model.train()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, device)
        print('Time taken: %.2f sec.' % (time.time() - start))

        model.eval()
        avg_test_acc, avg_loss = test(test_loader, model, criterion, optimizer, device, args, report=False)

        print('\nEvaluation:')
        print('\tVal Acc: %.2f - Loss: %.4f' % (avg_test_acc, avg_loss))
        print('\tCurrent best val acc: %.2f' % best_acc)

        # Log epoch to tensorboard
        # See log using: tensorboard --logdir='logs' --port=6006
        #util.logEpoch(logger, resnet, epoch + 1, avg_loss, avg_test_acc)


        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_loss)
        val_accuracies.append(avg_test_acc)

        np.save(args.log_dir + '/' + args.saved_fname + 'train_loss.npy',
                np.asarray(train_losses))
        np.save(args.log_dir + '/' + args.saved_fname + 'test_loss.npy',
                np.asarray(val_losses))
        np.save(args.log_dir + '/' + args.saved_fname + 'train_acc.npy',
                np.asarray(train_accuracies))
        np.save(args.log_dir + '/' + args.saved_fname + 'test_acc.npy',
                np.asarray(val_accuracies))


    print('Finished training!')
    # Save model
    print('Saving model - Acc: %.2f' % avg_test_acc)
    best_acc = avg_test_acc
    best_loss = avg_loss
    torch.save({
        'epoch': epoch + 1,
        #'state_dict': resnet.state_dict(),
        #'body': model.body.state_dict(),
        #'feat': model.head.state_dict(),
        'model': model.state_dict(),
        'acc': avg_test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict()
    }, os.path.join(args.log_dir, args.saved_fname+".pth.tar"))

    if args.save_to_pt:
        #model.eval()
        with torch.no_grad():
            example_x, _ = dset_train[0]
            example = torch.from_numpy(example_x).to('cuda:0')
            model.to('cuda:0')
            traced_model = torch.jit.trace(model, example.view(1, *example.size()))
            traced_model.save(args.log_dir + '/' + args.saved_fname + "_cuda.pt")

            example = example.to('cpu')
            model.to('cpu')
            traced_model = torch.jit.trace(model, example.view(1, *example.size()))
            traced_model.save(args.log_dir + '/' + args.saved_fname + "_cpu.pt")

            model.to(device)

    model.eval()
    print('Report on train set')
    _, _ = test(train_loader, model, criterion, optimizer, device, args, report=True, name='train')

    print('Report on test set')
    _, _ = test(test_loader, model, criterion, optimizer, device, args, report=True, name='test')



    LOG_FOUT.close()
    return


def log_string(out_str):
    LOG_FOUT.write(str(out_str)+'\n')
    LOG_FOUT.flush()
    log_string(out_str)


def load_checkpoint(args, model):
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    fname = os.path.join(args.ckpt_dir, args.ckpt_fname + '.pth.tar')
    assert os.path.isfile(fname), 'Error: no checkpoint file found!'

    checkpoint = torch.load(fname)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.body.load_state_dict(checkpoint['body'])
    model.head.load_state_dict(checkpoint['head'])

    return start_epoch, best_acc


def train(loader, model, criterion, optimizer, device):
    num_batch = len(loader)
    batch_size = loader.batch_size
    total = torch.LongTensor([0])
    correct = torch.LongTensor([0])
    total_loss = 0.

    #end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        #start = time.time()
        #print('Load time: ', start - end)
        #inputs = torch.from_numpy(inputs)
        inputs, targets = inputs.to(device), targets.to(device)
        # in 0.4.0 variable and tensor are merged
        #inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()

        # Do predictions
        if criterion.__class__ == nn.BCELoss:
            predicted = outputs.detach().round()
        else:
            _, predicted = torch.max(outputs.detach(), 1)

        total += batch_size
        correct += (predicted == targets).cpu().sum()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #log_iter = 1000
        #f (i + 1) % log_iter == 0:
        #  print("\tIter [%d/%d] Loss: %.4f" % (i + 1, num_batch, total_loss/log_iter))
        # total_loss = 0.

        #end = time.time()
        #print("Batch processing time:", end - start)

    loss = total_loss / (i+1)
    acc = 100. * correct.item() / total.item()
    print("Train Loss %.4f  Train Accuracy %.2f" % (loss, acc))

    return acc, loss


def test(loader, model, criterion, optimizer, device, args, report=False, name=None):
    # Eval
    total = torch.LongTensor([0])
    correct = torch.LongTensor([0])

    total_loss = 0.0
    n = 0

    all_targets = []
    all_predicted = []
    all_scores = []

    for i, (inputs, targets) in enumerate(loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            #inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n += 1
            # Do predictions
            if criterion.__class__ == nn.BCELoss:
                predicted = outputs.detach().round()
            else:
                _, predicted = torch.max(outputs.detach(), 1)

            total += targets.size(0)
            correct += (predicted == targets).cpu().sum()

            all_targets.append(targets.cpu())
            all_predicted.append(predicted.cpu())
            all_scores.append(outputs.detach().cpu())

    if report:
        targets = torch.cat(all_targets, 0).numpy()
        predicted = torch.cat(all_predicted, 0).numpy()
        scores = torch.cat(all_scores, 0).numpy()

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, predicted).ravel()

        # ROC curve
        fname = args.log_dir + '/' + args.saved_fname + '_' + name
        # pr, tpr = plot_roc(targets, scores, plot_name)

        fpr, tpr, _ = roc_curve(targets, scores)
        roc_data = np.stack((fpr, tpr), axis=0)
        np.save(fname + '_roc.npy', roc_data)
        np.save(fname + '_scores.npy', scores)
        np.save(fname + '_targets.npy', targets)

        # n_positives = targets.cpu().sum().numpy().flatten()[0]
        #    n_negatives = total.cpu().numpy()[0] - n_positives


        print('True positives: %d' % tp)
        print('True negatives: %d' % tn)
        print('False positives: %d' % fp)
        print('False negatives: %d' % fn)

        print("Classification report")

        print(classification_report(targets, predicted))

    avg_test_acc = 100. * correct.item() / total.item()
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss

def plot_roc(targets, scores, name):

    # ROC curve
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc =  auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_fname', type=Path, help='training .tar file or directory')
    parser.add_argument('--testing_fname', type=Path, help='testing .tar file or directoy')
    parser.add_argument('--rope_data', action='store_true', help='Use data for rope with robot grippers')
    parser.add_argument('--model', default='voxnet', help='Model name: [default:voxnet]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--num_classes', type=int, default=40, help='Category Number [10/30/40] [default: 40]')
    parser.add_argument('--max_epoch', type=int, default=256, help='Epoch to run [default: 256]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=16, help='Decay step for lr decay [default: 16]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--saved_fname', type=Path, default=None, help='name of weight to be saved')
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--ckpt_dir', default='log', help='check point dir [default: log]')
    parser.add_argument('--ckpt_fname', default='model', help='check point name [default: model]')
    parser.add_argument('--num_channels', default=3, type=int, help='number of channels for voxel grid')
    parser.add_argument('--use_same_net', action='store_true', help='For multiple channel voxel grid, use the same '
                                                                    'network for each channel?')
    parser.add_argument('--save_to_pt', action='store_true', help='save to pt for loading with libtorch')
    args = parser.parse_args()
    #cudnn.benchmark = True
    main(args)
