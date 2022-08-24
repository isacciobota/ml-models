import torch
from torch import nn
import torchvision
from torch._C._te import simplify
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import onnx
from onnx import shape_inference

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


def load_fakeData(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    #download data
    data_train = torchvision.datasets.FakeData(transform=trans)
    data_test = torchvision.datasets.FakeData(transform=trans)
    #split data
    data_train, data_val = torch.utils.data.random_split(data_train, [300, 700], generator=torch.Generator().manual_seed(42))
    #return data
    return (torch.utils.data.DataLoader(data_train, batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(data_val, batch_size, shuffle=False, num_workers=2),
            torch.utils.data.DataLoader(data_test, batch_size, shuffle=False, num_workers=2))


def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += float(l)
            total_hits += sum(net(X).argmax(axis=1).type(y.dtype) == y)
            total_samples += y.numel()
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100


def train_epoch(net, train_iter, loss, optimizer, device):
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += float(l)
        total_hits += sum(y_hat.argmax(axis=1).type(y.dtype) == y)
        total_samples += y.numel()
    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples  * 100


def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # print('Training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        # print(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}')
    test_loss, test_acc = evaluate_accuracy(net, test_iter, loss, device)
    # print(f'Test loss {test_loss:.2f}, Test accuracy {test_acc:.2f}')

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all, test_acc


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def plot_accuracy(train_acc_all, val_acc_all):
    epochs = range(1, len(train_acc_all) + 1)
    plt.plot(epochs, train_acc_all, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_all, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


class MobileNets(nn.Module):
    def __init__(self, ch_in, n_classes, alpha=1):
        super(MobileNets, self).__init__()

        def use_alpha(inp):
            return int(inp*alpha)

        def conv_block(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_depthwise_block(inp, oup, stride):
            # depthwise separable convolutions is a form of factorized convolutions which factorize a standard convolution
            # into a depthwise convolution and a 1x1 convolution called a pointwise convolution
            return nn.Sequential(
                # depthwise
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pointwise
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_block(ch_in, use_alpha(32), 2),
            conv_depthwise_block(use_alpha(32), use_alpha(64), 1),
            conv_depthwise_block(use_alpha(64), use_alpha(128), 2),
            conv_depthwise_block(use_alpha(128), use_alpha(128), 1),
            conv_depthwise_block(use_alpha(128), use_alpha(256), 2),
            conv_depthwise_block(use_alpha(256), use_alpha(256), 1),
            conv_depthwise_block(use_alpha(256), use_alpha(512), 2),
            conv_depthwise_block(use_alpha(512), use_alpha(512), 1),
            conv_depthwise_block(use_alpha(512), use_alpha(512), 1),
            conv_depthwise_block(use_alpha(512), use_alpha(512), 1),
            conv_depthwise_block(use_alpha(512), use_alpha(512), 1),
            conv_depthwise_block(use_alpha(512), use_alpha(512), 1),
            conv_depthwise_block(use_alpha(512), use_alpha(1024), 2),
            conv_depthwise_block(use_alpha(1024), 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def display_summary(alpha, resolution_multiplier):
    batch_size, lr, num_epochs = 32, 0.05, 5
    image_size = int(224*resolution_multiplier)
    train_iter, val_iter, test_iter = load_fakeData(batch_size, resize=image_size)

    model = MobileNets(ch_in=3, n_classes=10, alpha=alpha)
    s = summary(model, input_size=(3, 224, 224), device='cpu', verbose=0)
    train_loss_all, train_acc_all, val_loss_all, val_acc_all, test_acc = train(model, train_iter, val_iter, test_iter, num_epochs, lr, try_gpu())
    format_test_acc = "{:.2f}".format(test_acc)
    print(alpha, "MobileNet", image_size, "• Total params:", s.total_params, "• Test accuracy:", format_test_acc)

def prepare_onnx():
    dummy_input = torch.randn(1, 3, 224, 224)
    model = MobileNets(ch_in=3, n_classes=10)
    torch.onnx.export(model, dummy_input, "mobilenetsx.onnx", verbose=True)
    model = onnx.load("mobilenetsx.onnx")
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, "mobilenets.onnx")

if __name__ == '__main__':
    alpha = [1, 0.75, 0.5, 0.25]
    resolution_multiplier = [1, 0.86, 0.715, 0.575]

    for alpha in alpha:
        display_summary(alpha, 1)

    print("=====================================")

    for resolution in resolution_multiplier:
        display_summary(1, resolution)

    # prepare_onnx()