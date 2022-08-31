import torch
import torchvision
from onnx import shape_inference
from torchvision import transforms
from torch import nn, Tensor
import onnx
from onnxsim import simplify


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
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples * 100


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
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples * 100


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
    print('Training on', device)
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
        print(
            f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}')
    test_loss, test_acc = evaluate_accuracy(net, test_iter, loss, device)
    print(f'Test loss {test_loss:.2f}, Test accuracy {test_acc:.2f}')

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def load_data_fakedata(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fakedata_train = torchvision.datasets.FakeData(
        transform=trans)
    fakedata_test = torchvision.datasets.FakeData(
        transform=trans)
    fakedata_train, fakedata_val = torch.utils.data.random_split(fakedata_train, [300, 700],
                                                                 generator=torch.Generator().manual_seed(42))
    return (torch.utils.data.DataLoader(fakedata_train, batch_size, shuffle=True,
                                        num_workers=2),
            torch.utils.data.DataLoader(fakedata_val, batch_size, shuffle=False,
                                        num_workers=2),
            torch.utils.data.DataLoader(fakedata_test, batch_size, shuffle=False,
                                        num_workers=2))


class ConvBlock(nn.Module):
    # Convolution Block with Conv2d layer, Batch Normalization and ReLU. Act is an activation function.
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            act=nn.ReLU(),
            groups=1,
            bn=True,
            bias=False
    ):
        super().__init__()

        # If k = 1 -> p = 0, k = 3 -> p = 1, k = 5, p = 2.
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))


class SeBlock(nn.Module):
    # Squeeze and Excitation Block.
    def __init__(
            self,
            in_channels: int
    ):
        super().__init__()

        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [N, C, H, W].
        f = self.globpool(x)
        f = torch.flatten(f, 1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:, :, None, None]
        # f shape: [N, C, 1, 1]

        scale = x * f
        return scale


# BNeck
class BNeck(nn.Module):
    # MobileNetV3 Block
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            exp_size: int,
            se: bool,
            act: torch.nn.modules.activation,
            stride: int
    ):
        super().__init__()

        self.add = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_size, 1, 1, act),
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se == True else nn.Identity(),
            ConvBlock(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.add:
            res = res + x

        return res


""" MobileNetV3 """


class MobileNetV3(nn.Module):
    def __init__(self,
                 config_name: str,
                 in_channels=3,
                 classes=10):

        super().__init__()
        config = self.config(config_name)

        # First convolution(conv2d) layer.
        self.conv = ConvBlock(in_channels, 16, 3, 2, nn.Hardswish())
        # Bneck blocks in a list.
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))

        # Classifier
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, classes, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)

        x = self.classifier(x)
        return torch.flatten(x, 1)

    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)]
        large = [
            [3, 16, 16, 16, False, RE, 1],
            [3, 64, 16, 24, False, RE, 2],
            [3, 72, 24, 24, False, RE, 1],
            [5, 72, 24, 40, True, RE, 2],
            [5, 120, 40, 40, True, RE, 1],
            [5, 120, 40, 40, True, RE, 1],
            [3, 240, 40, 80, False, HE, 2],
            [3, 200, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 184, 80, 80, False, HE, 1],
            [3, 480, 80, 112, True, HE, 1],
            [3, 672, 112, 112, True, HE, 1],
            [5, 672, 112, 160, True, HE, 2],
            [5, 960, 160, 160, True, HE, 1],
            [5, 960, 160, 160, True, HE, 1]
        ]

        small = [
            [3, 16, 16, 16, True, RE, 2],
            [3, 72, 16, 24, False, RE, 2],
            [3, 88, 24, 24, False, RE, 1],
            [5, 96, 24, 40, True, HE, 2],
            [5, 240, 40, 40, True, HE, 1],
            [5, 240, 40, 40, True, HE, 1],
            [5, 120, 40, 48, True, HE, 1],
            [5, 144, 48, 48, True, HE, 1],
            [5, 288, 48, 96, True, HE, 2],
            [5, 576, 96, 96, True, HE, 1],
            [5, 576, 96, 96, True, HE, 1]
        ]

        if name == "large": return large
        if name == "small": return small


if __name__ == "__main__":
    name = "small"
    rho = 1
    res = int(rho * 224)
    num_epochs, lr, batch_size = 5, 0.1, 32
    train_iter, val_iter, test_iter = load_data_fakedata(batch_size)

    net = MobileNetV3(name)
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, test_iter, num_epochs,
                                                                     lr, try_gpu())

    randim = torch.rand(1, 3, res, res)
    print(net(randim).shape)
    torch.onnx.export(net, randim, "mobilenetv3_small.onnx", input_names=['input'], output_names=['output'])
    model = onnx.load("mobilenetv3_small.onnx")
    inferred_model = shape_inference.infer_shapes(model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, "mobilenetv3_small_simplified.onnx")
