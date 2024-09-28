import jittor as jt
import numpy as np

def clear_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.stop_grad()
            p.grad.zero_()


def compute_loss_accuracy(net, data_loader, criterion):
    net.eval()
    correct = 0
    total_loss = 0.0

    with jt.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).stop_grad().item()
            pred = np.argmax(outputs.stop_grad(), -1)
            correct += (pred == labels.data).astype(float).sum()

    return total_loss / (batch_idx + 1), 100.0 * (correct / len(data_loader))
