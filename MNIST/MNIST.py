import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import trange
import datetime
import os

import matplotlib.pyplot as plt

# 数据预处理：转换为Tensor并进行归一化
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# 下载并加载训练数据
trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)


def modelTrain():
    # 定义模型
    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))

    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    # 训练模型
    epochs = 5
    losses = []  # List to store the training loss for each epoch
    for e in range(epochs):
        running_loss = 0
        with trange(len(trainloader)) as t:
            t.set_description(f'Epoch {e+1}/{epochs}')
            for i, (images, labels) in enumerate(trainloader):
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                # Training pass
                optimizer.zero_grad()

                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t.set_postfix(loss=running_loss / (i + 1))
                t.update()

        epoch_loss = running_loss / len(trainloader)
        losses.append(epoch_loss)
        print(f"Training loss: {epoch_loss}")

    # 保存模型
    save_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(
        save_dir, f'mnist_model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
    torch.save(model, model_path)

    # 删除最旧的模型，如果模型数量超过5个
    model_files = sorted(os.listdir(save_dir))
    if len(model_files) > 5:
        oldest_model = os.path.join(save_dir, model_files[0])
        os.remove(oldest_model)

    # Plotting the training loss
    plt.plot(range(1, epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

    return model


if __name__ == '__main__':
    # 加载模型, 留空为新训练
    model = 'MNIST/model/mnist_model_2024-03-06_17-10-31.pth'
    if model == '':
        model = modelTrain()
    else:
        model = torch.load(model)

    # 随机选择n个样本
    n = 15  # 修改n的值为你想要的样本数量
    sample_indices = torch.randint(len(trainset), (n,))
    sample_images = torch.stack([trainset[i][0] for i in sample_indices])
    sample_labels = torch.tensor([trainset[i][1] for i in sample_indices])

    # 将样本输入模型进行预测
    with torch.no_grad():
        sample_images = sample_images.view(sample_images.shape[0], -1)
        sample_output = model(sample_images)
        sample_predictions = torch.argmax(sample_output, dim=1)

    # 展示手写图像与识别结果
    fig, axes = plt.subplots(n // 5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i].view(28, 28), cmap='gray')
        ax.set_title(
            f'Prediction: {sample_predictions[i]}, Label: {sample_labels[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
