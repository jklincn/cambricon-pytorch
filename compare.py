import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch_mlu
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

num_epochs = 5

lr_single = 0.001
batch_size_single = 32

lr_multi = 0.001
batch_size_multi = 32

device = "mlu"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("cncl", rank=rank, world_size=world_size)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.mlu(), labels.mlu()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    return accuracy


def multi_train(rank, world_size, losses, accuracy_storage):
    setup(rank, world_size)

    torch.mlu.set_device(rank)

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据集
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_multi, sampler=train_sampler, num_workers=2
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_multi, shuffle=False, num_workers=2
    )

    # 加载预训练的 mobilenet_v2 模型并修改最后的分类层
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_multi, momentum=0.9)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if rank == 0:
                losses.append(loss.item())
            if i % 100 == 0:
                if rank == 0:  # 仅在主进程中打印
                    print(
                        f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {loss.item():.4f}"
                    )

    if rank == 0:  # 仅在 rank 0 上进行评估并存储准确率
        accuracy = evaluate(model, test_loader)
        accuracy_storage.append(accuracy)

    dist.destroy_process_group()


def single_card():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_single, shuffle=True, num_workers=2
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_single, shuffle=False, num_workers=2
    )
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_single, momentum=0.9)

    start_time = time.time()
    single_losses = []
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            single_losses.append(loss.item())
            if i % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {loss.item():.4f}"
                )
    accuracy = evaluate(model, test_loader)
    end_time = time.time()

    return accuracy, single_losses, end_time - start_time


def multi_card():
    world_size = torch.mlu.device_count()
    start_time = time.time()
    with mp.Manager() as manager:
        multi_losses = manager.list()  # 创建一个共享的损失列表
        accuracy_storage = manager.list()  # 创建一个共享的准确率列表
        mp.spawn(
            multi_train,
            args=(world_size, multi_losses, accuracy_storage),
            nprocs=world_size,
        )
        multi_losses = list(multi_losses)
        accuracy = accuracy_storage[0]
    end_time = time.time()
    return accuracy, multi_losses, end_time - start_time


def plot_metrics(
    single_losses,
    multi_losses,
    single_time,
    multi_time,
    single_accuracy,
    multi_accuracy,
):
    plt.figure(figsize=(16, 12))

    # Loss comparison
    plt.subplot(2, 2, 1)
    plt.plot(single_losses, label="Single Card Training Loss")
    plt.plot(multi_losses, label="Multi Card Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()

    # Loss comparison(smooth)
    def moving_average(data, window_size):
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        ma_vec = np.insert(ma_vec, 0, [ma_vec[0]] * (window_size - 1))
        return ma_vec

    window_size = 50
    smoothed_single_losses = moving_average(single_losses, window_size)
    smoothed_multi_losses = moving_average(multi_losses, window_size)
    plt.subplot(2, 2, 2)
    plt.plot(smoothed_single_losses, label="Single Card Training Loss", color="blue")
    plt.plot(smoothed_multi_losses, label="Multi Card Training Loss", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Smoothed Training Loss Comparison with Shaded Areas")
    plt.legend()

    # Time comparison
    plt.subplot(2, 2, 3)
    indices = np.arange(2)
    bar_width = 0.35
    bars = plt.bar(indices, [single_time, multi_time], bar_width, label="Training Time")
    plt.xlabel("Training Type")
    plt.ylabel("Time (s)")
    plt.title("Training Time Comparison")
    plt.xticks(indices, ["Single Card", "Multi Card"])
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            round(yval, 2),
            ha="center",
            va="bottom",
        )

    # Accuracy comparison
    plt.subplot(2, 2, 4)
    bars = plt.bar(
        indices, [single_accuracy, multi_accuracy], bar_width, label="Accuracy"
    )
    plt.xlabel("Training Type")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.xticks(indices, ["Single Card", "Multi Card"])
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            round(yval, 2),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("training_comparison.png")
    plt.show()


if __name__ == "__main__":
    if not torch.mlu.is_available():
        sys.exit("MLU is not available")
    if torch.mlu.device_count() == 1:
        sys.exit("There is only one MLU")
    single_accuracy, single_losses, single_time = single_card()
    multi_accuracy, multi_losses, multi_time = multi_card()
    plot_metrics(
        single_losses,
        multi_losses,
        single_time,
        multi_time,
        single_accuracy,
        multi_accuracy,
    )
