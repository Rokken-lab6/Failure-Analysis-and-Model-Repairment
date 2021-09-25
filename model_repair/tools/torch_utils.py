import torch
import matplotlib.pyplot as plt
import torchvision

functions = ["t2n", "plot", "plot_old", "info", "plot_c"]

ssh_mode = False
def t2n(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x


def plot(x, normalize=False):
    # x: ..., H ,W
    x = x.view(-1, 1, *x.shape[-2:])
    B = x.shape[0]
    img = torchvision.utils.make_grid(x, nrow=int(B ** (1 / 2)), normalize=normalize)
    img = img.permute(1, 2, 0)
    plt.imshow(img.t2n(), interpolation="none")
    # plt.show()
    display()


def plot_old(x):
    plt.imshow(x.squeeze().t2n())
    # plt.show()
    display()

def plot_c(x):
    if len(x.shape) == 3:
        if x.shape[0] == 3:
            x = x.permute(1, 2, 0)

    elif len(x.shape) == 4:
        if x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        # TODO: display more than one
        print(f"Displaying only first image in batch of {x.shape[0]}.")
        x = x[0]

    plt.imshow(x.squeeze().t2n())
    # plt.show()
    display()

def info(x):
    print(f"Shape: {list(x.shape)}")
    print(f"Names: {x.names}")
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.float().mean()}")
    print(f"Dtype: {x.dtype}, Device: {x.device}")

    if x.dtype in [torch.bool]:
        x = x.long()

    try:
        uniques = torch.unique(x).tolist()
        if len(uniques) <= 10:
            print(f"Unique: {uniques}, Count: {len(uniques)}")
        else:
            print(f"Unique[0:10]: {uniques[0:10]}, Count: {len(uniques)}")
    except:
        print("Unique() exception")

def display():
    if not ssh_mode:
        plt.show()
    else:
        plt.savefig("current_plot.png")
        # input()

for f in functions:
    setattr(torch.Tensor, f, locals()[f])

# Dirty but useful
torch.Tensor.i = property(lambda self: self.info())
torch.Tensor.p = property(lambda self: self.plot())
torch.Tensor.np = property(lambda self: self.t2n())
torch.Tensor.pc = property(lambda self: self.plot_c())
