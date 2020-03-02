import torch
from torchvision import datasets, transforms
import pathlib
import os
import argparse
import models

# Train the given net on the data
def train(args, net, data, device, optimizer, loss_fn):
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, .985, last_epoch=-1)

    net.train()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_smloss, train_error, train_cert = 0., 0., 0., 0.
        n = 0
        for x, y in data['train']:
            x, y = x.to(device), y.to(device)

            n += y.shape[0]
            optimizer.zero_grad()
            pred = net(x)

            # Compute classification loss
            smloss = loss_fn(pred, y)

            # Compute Lipshcitz constant for each logit
            lc = net.get_lc()
            real_lc = lc

            loss = smloss + args.lc_penalty*torch.sum(lc*torch.nn.functional.softmax(lc, dim=0))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_smloss += smloss.item()
            train_error += (pred.argmax(dim=1)!=y).sum()
            train_cert += compute_cert(pred, y, real_lc, cert_level=args.cert_magnitude)

        train_loss /= len(data['train'])
        train_smloss /= len(data['train'])
        train_error /= n
        train_cert /= n

        train_cert = 1-train_cert
        sched.step(epoch)

        real_lc = torch.max(lc).item()
        print(
            'epoch: {}/{}, train loss: {:.3f}, lc:{:.3f}, class loss:{:.3f}, error:{:.3f}, cert error {:.3f}'.format(
                epoch, args.epochs, train_loss, real_lc, train_smloss,train_error,train_cert))


# Computes worst case adversarial accuracy for perturbations of size "cert_level"
def compute_cert(pred, y, lc, cert_level=0.1):

    # More complicated, but slightly better certificate
    # true_class_score = torch.stack([pred[i,y[i]] for i in range(y.shape[0])])
    # true_class_lc = lc[y]
    # false_class_scores = torch.stack([torch.cat((pred[i, :y[i]], pred[i, (y[i]+1):])) for i in range(y.shape[0])])
    # false_class_lcs = torch.stack([torch.cat((lc[:y[i]], lc[(y[i] + 1):])) for i in range(y.shape[0])])
    #
    # margins = true_class_score.reshape(-1,1) - false_class_scores - \
    #           cert_level * (true_class_lc.reshape(-1,1) + false_class_lcs)
    # certs = (margins>0).all(dim=1).sum().item()

    # Compute a Lipschitz constant for the margin (2x that of the logits)
    lc = 2*torch.max(lc)

    # Compute margin for top answer
    sorted = pred.sort(1, descending=True)
    margin = sorted[0][:, 0] - sorted[0][:, 1]

    # Score predictions
    _, y_cls = pred.max(dim=1)

    # Combine accuracy and margins
    certs = (y_cls.eq(y) & (margin > (cert_level * torch.max(lc)))).sum().item()

    return certs


# Evaluate the net on the data
def evaluate(args, net, data, device):
    net.eval()
    report = {'nb_test': 0, 'correct': 0, 'correct_fgm': 0, 'correct_pgd': 0, 'certified': 0}
    for x, y in data['test']:
        x, y = x.to(device), y.to(device)

        output = net(x)
        _, y_cls = output.max(1)
        lc = net.get_lc()
        ncert = compute_cert(output, y, lc, cert_level=args.cert_magnitude)

        report['nb_test'] += y.size(0)
        report['correct'] += y_cls.eq(y).sum().item()
        report['certified'] +=ncert

    print('test acc on clean examples (%): {:.3f}'.format(report['correct'] / report['nb_test'] * 100.))
    print('test certified on clean examples (%): {:.3f}'.format(report['certified'] / report['nb_test'] * 100.))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Lipschitz Certification Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training and testing (default: 128)')
    parser.add_argument('--lr', type=float, default=1E-3, metavar='LR',
                        help='learning rate (default: 1E-3)')
    parser.add_argument('--gamma', type=float, default=0.985, metavar='M',
                        help='Learning rate step gamma (default: 0.985)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lc_penalty', type=float, default=0.01,
                        help='Penalty weight for Lipschitz constant (default: 0.01)')
    parser.add_argument('--cert_magnitude', type=float, default=0.1,
                        help='Perturbation magnitude used to compute certificate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--group_size', type=int, default=10,
                        help='Group size for group sort activation (default: 10)')
    parser.add_argument('--activation', type=str, default='groupsort',
                        help='Activation function; either groupsort or relu')
    parser.add_argument('--save-fn', type=str, default='mnist_groupsort.p',
                        help='Filename to save the trained model')
    parser.add_argument('--save-dir', type=str, default='outputs',
                        help='Directory to save the trained model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    n_workers = 1 if use_cuda else 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load training and test data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (1,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (1,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=n_workers)

    # Instantiate model, loss, and optimizer for training
    if args.activation == 'groupsort':
        activation = lambda: models.GroupSort(group_size=args.group_size)
    elif args.activation == 'relu':
        activation = lambda: models.ReLU_LC()
    else:
        raise Exception("Activation must be groupsort or relu")

    net = models.ShallowNet(activation=activation)
    net.apply(models.lc_init)

    if use_cuda:
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    data = {'train': train_loader, 'test': test_loader}
    train(args, net, data, device, optimizer, loss_fn)

    # Evaluate on test data
    evaluate(args, net, data, device)

    # Save the model for future use
    save_path = os.path.join(args.save_dir, args.save_fn)
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()