import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt

from video_dataset import Dataset_3DCNN
from CNN3D import CNN3D
from sklearn.metrics import accuracy_score

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)
    
     # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fc_hidden",
        type=tuple,
        nargs="?",
        default= (256,256),
        help="fc hidden layer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        nargs="?",
        default= 0.0,
        help="sampling method"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="?",
        help="num of class",
        default= 6,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?",
        default= 15,
        help="epoch number"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default= 10,
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="learning rate",
        default= 1e-4,
    )
    parser.add_argument(
        "--log",
        type=int,
        nargs="?",
        help="log interval",
        default= 5,
    )
    parser.add_argument(
        "--frame_size",
        type=tuple,
        nargs="?",
        help="resize of frame",
        default= (256,342),
    )
    parser.add_argument(
        "--sp_rate",
        type=int,
        nargs="?",
        help="sampling rate",
        default= 20,
    )
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    # set path
    train_video_path = "data/train/"    
    train_label_path = "data/train.txt"  
    valid_video_path = "data/validate"
    valid_label_path = "data/validate.txt"  
    save_model_path = "outputs/checkpoints"  # save Pytorch models
    
    opt = parse_args()
    print('------Loading data and model------')
    # data loading parameters
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    params = {'batch_size': opt.batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # image transformation
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean = [0.07], std = [0.092])])
    # dataloader
    train_set = Dataset_3DCNN(train_video_path, train_label_path, opt.sp_rate, opt.frame_size, transform = transform)
    valid_set = Dataset_3DCNN(valid_video_path, valid_label_path, opt.sp_rate, opt.frame_size, transform = transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # create model
    cnn3d = CNN3D(t_dim = opt.sp_rate, img_x = (opt.frame_size)[0], img_y = (opt.frame_size)[1],
                drop_p=opt.dropout, fc_hidden1 = (opt.fc_hidden)[0],  fc_hidden2 = (opt.fc_hidden)[1], num_classes = opt.k).to(device)

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=opt.lr)   # optimize all cnn parameters

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    print('------Training begin------')
    # start training
    for epoch in range(opt.epochs):
        # train, test model
        train_losses, train_scores = train(opt.log, cnn3d, device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation(cnn3d, device, optimizer, valid_loader)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
        np.save('outputs/3DCNN_epoch_training_losses.npy', A)
        np.save('outputs/3DCNN_epoch_training_scores.npy', B)
        np.save('outputs/3DCNN_epoch_test_loss.npy', C)
        np.save('outputs/3DCNN_epoch_test_score.npy', D)

    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, opt.epochs + 1), A[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, opt.epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, opt.epochs + 1), B[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, opt.epochs + 1), D)         #  test accuracy (on epoch end)
    # plt.plot(histories.losses_val)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc="upper left")
    title = "outputs/fig_3DCNN.png"
    plt.savefig(title, dpi=600)
    plt.close(fig)

                            
