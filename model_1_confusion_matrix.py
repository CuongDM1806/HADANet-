import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

title = ""

# 定义双非共享注意力块
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.attention_weights = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def forward(self, x):
        feature = F.relu(self.conv2(self.conv1(x)))
        score = F.softmax(feature, dim=2)
        # attention_map = torch.sigmoid(self.conv(x))  # 用卷积生成注意力图
        # print(attention_map)
        weighted_features = x * score  # 加权特征
        return weighted_features


# 定义特征提取网络（CNN）
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=64):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.attention_block = AttentionBlock(64)
        self.fc1 = nn.Linear(64 * 5 * 4, 1000)  # Flatten后接全连接层
        self.fc2 = nn.Linear(1000, 100)  # Flatten后接全连接层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # print("x.shape:", x.shape)  # x.shape torch.Size([64, 64, 5, 4])
        x = F.relu(self.bn1(self.conv1(x)))
        # print("x.shape经过conv1:", x.shape)  # torch.Size([64, 64, 5, 4])
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)  # torch.Size([64, 256, 5, 4])
        # x = self.attention_block(x)
        # print(x.shape)  # torch.Size([64, 256, 5, 4])
        x = x.view(x.size(0), -1)  # Flatten
        # print("x.shape:", x.shape)  # torch.Size([64, 5120])
        x = F.relu(self.fc2(self.fc1(x)))
        # print("x.shape经过fc:", x.shape)  # torch.Size([64, 100])
        return x

# 定义H-CNN 输入 [batch_size, in_channel, 5, 4]
class H_CNN(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, stride=1, padding=0):
        super(H_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 4))
        self.bn = nn.BatchNorm2d(out_channel)
        # self.fc1 = nn.Linear(64 * 5 * 4, 1000)  # Flatten后接全连接层
        # self.fc2 = nn.Linear(1000, 100)  # Flatten后接全连接层

    def forward(self, x):
        # print("x.shape:", x.shape)
        x1 = self.conv1(x)
        # print("x1.shape:", x1.shape)
        x2 = self.conv2(x)
        # print("x2.shape:", x2.shape)
        x = x2 * x1
        x = F.relu(self.bn(x))
        x = x.view(x.size(0), -1)  # Flatten
        # x = F.relu(self.fc2(self.fc1(x)))
        return x    # 输出也得是[batch_size, out_channel, 5, 4]

# 定义MLP分类器
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 定义网络的层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层
        self.relu = nn.ReLU()  # 激活函数（这里使用ReLU）
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播过程
        x = self.fc1(x)  # 输入经过第一个全连接层
        x = self.dropout(x)
        x = self.relu(x)  # 应用激活函数
        x = self.fc2(x)  # 隐藏层输出到输出层
        x = self.dropout(x)
        x = self.softmax(x)  # 使用Softmax输出概率分布
        return x

# 定义域适应性模块（通过最大均值差异（MMD）减少源域和目标域的分布差异）
class DomainAdaptationModule(nn.Module):
    def __init__(self):
        super(DomainAdaptationModule, self).__init__()

    def forward(self, source_features, target_features):
        # 计算最大均值差异（MMD）
        source_mean = source_features.mean(dim=0)
        target_mean = target_features.mean(dim=0)
        mmd_loss = torch.sum((source_mean - target_mean) ** 2)
        return mmd_loss


# 定义完整的模型（包含特征提取、注意力机制、域适应、分类器）
class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = H_CNN()
        self.domain_adaptation_module = DomainAdaptationModule()
        # self.classifier = nn.Linear(100, 4)  # 4分类任务
        self.mlp = MLP(1280, 640, 4)

    def forward(self, source_data, target_data):
        source_features = self.feature_extractor(source_data)
        target_features = self.feature_extractor(target_data)

        # 域适应性损失
        domain_loss = self.domain_adaptation_module(source_data, target_data)

        # 分类任务
        # source_classification = F.softmax(self.classifier(source_features), dim=1)
        # target_classification = self.classifier(target_features)
        source_classification = self.mlp(source_features)
        target_classification = self.mlp(target_features)

        return source_classification, target_classification, #domain_loss


# 训练函数
def train(model, train_loader, optimizer, criterion, domain_criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for source_data, target_data, source_labels in train_loader:
        source_data, target_data, source_labels = source_data.to(device), target_data.to(device), source_labels.to(
            device)

        optimizer.zero_grad()

        # 正向传播
        source_classification, target_classification  = model(source_data, target_data) #domain_loss

        # 分类损失（交叉熵）
        # print(source_classification.dtype)  # torch.float32
        # print(source_labels.dtype)  # torch.float32
        source_labels = source_labels.type(torch.long)
        # print(source_labels.dtype)  # torch.int64
        classification_loss = criterion(source_classification, source_labels)

        # 总损失 = 分类损失 + 域适应性损失
        total_loss = classification_loss# + domain_loss
        total_loss.backward()

        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(source_classification, 1)
        correct += (predicted == source_labels).sum().item()
        total += source_labels.size(0)
        total_loss += total_loss.item()

    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy


# 预测函数
def predict(model, test_data, device):
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        features = model.feature_extractor(test_data)
        # predictions = model.classifier(features)
        output = model.mlp(features)
        _, predicted = torch.max(output, 1)
        return predicted
        # return torch.argmax(predictions, dim=1)


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for num in range(8, 9):
#
#     # 从特征文件夹和标签文件夹加载数据
#     feature_folder = "D:\postgraduate\MIdata\physionet\libo\physionet\DE/"
#     label_folder = "D:\postgraduate\MIdata\physionet\libo\physionet\label/"
#
#     # 从特征文件夹中加载数据
#     features = []
#     test_features = []
#     for i in range(1, 91):
#         if i == num:
#             test_feature_file = np.load(feature_folder + "S{}.npy".format(i))
#             test_features.append(test_feature_file)
#             continue
#         feature_file = np.load(feature_folder + "S{}.npy".format(i))
#         features.append(feature_file)
#
#     # 从标签文件夹中加载数据
#     labels = []
#     test_labels = []
#     for i in range(1, 91):
#         if i == num:
#             test_label_file = np.load(label_folder + "S{}.npy".format(i))
#             test_labels.append(test_label_file)
#             continue
#         label_file = np.load(label_folder + "S{}.npy".format(i))
#         labels.append(label_file)
#
#     features = np.array(features)
#     labels = np.array(labels)
#     features = features.reshape(-1, 90, 4, 64, 5)
#     features = np.transpose(features, (0, 1, 3, 4, 2))
#     features = features.reshape((-1, 64, 5, 4))
#     labels = labels.reshape((-1, 90, 4, 1)).mean(axis=2)
#     labels = labels.reshape(-1)
#
#     test_features = np.array(test_features)
#     test_labels = np.array(test_labels)
#     test_features = test_features.reshape(-1, 90, 4, 64, 5)
#     test_features = np.transpose(test_features, (0, 1, 3, 4, 2))
#     test_features = test_features.reshape((-1, 64, 5, 4))
#     test_labels = test_labels.reshape((-1, 90, 4, 1)).mean(axis=2)
#     test_labels = test_labels.reshape(-1)
#
#     features = torch.tensor(features, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.float32)
#     test_features = torch.tensor(test_features, dtype=torch.float32)
#     test_labels = torch.tensor(test_labels, dtype=torch.float32)
#     # print(features.shape)  # torch.Size([8010, 64, 5, 4])
#     # print(labels.shape)  # torch.Size([8010])
#     # print(test_features.shape)  # torch.Size([90, 64, 5, 4])
#     # print(test_labels.shape)  # torch.Size([90])

best_accuracy_list = []
for num in range(48, 49):

    # 从特征文件夹和标签文件夹加载数据
    feature_folder = "D:\postgraduate\MIdata\physionet\libo\physionet\DE/"
    label_folder = "D:\postgraduate\MIdata\physionet\libo\physionet\label/"

    # 从特征文件夹中加载数据
    features = []
    for i in range(num, num+1):
        feature_file = np.load(feature_folder + "S{}.npy".format(i))
        features.append(feature_file)

    # 从标签文件夹中加载数据
    labels = []
    for i in range(num, num+1):
        label_file = np.load(label_folder + "S{}.npy".format(i))
        labels.append(label_file)

    features = np.array(features)
    labels = np.array(labels)
    # print(labels)
    # print(features.shape)  # (1, 360, 64, 5)
    # print(labels.shape)  # (1, 360, 1)
    # print(labels)
    features = features.reshape(-1, 90, 4, 64, 5)
    features = np.transpose(features, (0, 1, 3, 4, 2))
    features = features.reshape((-1, 64, 5, 4))
    # print(features.shape)  # (90, 64, 5, 4)
    labels = labels.reshape((-1, 90, 4, 1)).mean(axis=2)
    labels = labels.reshape(-1)
    # print(labels)
    # print(labels.shape)  # (90,)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    print(features.shape)  # torch.Size([90, 64, 5, 4])
    print(labels.shape)  # torch.Size([90])

    # 训练集、测试集划分
    # 创建一个索引列表
    indices = torch.arange(features.size(0))

    # 初始化训练集和测试集的索引列表
    train_indices = []
    test_indices = []

    for i in range(0, len(indices), 6):  # 每6个样本一组
        # 先取5个作为训练集
        for j in range(i, min(i + 5, len(indices))):
            train_indices.append(indices[j])
        # 然后取1个作为测试集
        if i + 5 < len(indices):
            test_indices.append(indices[i + 5])

    # 将训练集和测试集的索引转为 tensor
    train_indices = torch.tensor(train_indices)
    test_indices = torch.tensor(test_indices)

    # 根据索引划分数据
    features_train = features[train_indices]
    features_test = features[test_indices]
    labels_train = labels[train_indices]
    labels_test = labels[test_indices]

    x_train = features_train.detach()
    y_train = labels_train.detach()
    x_test = features_test.detach()
    y_test = labels_test.detach()
    print(x_train.shape)  # torch.Size([8010, 64, 5, 4])
    print(y_train.shape)  # torch.Size([8010])
    print(x_test.shape)  # torch.Size([90, 64, 5, 4])
    print(y_test.shape)  # torch.Size([90])
    print(y_test)

    # 数据加载
    train_dataset = TensorDataset(x_train, x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=15, drop_last=True, shuffle=True)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 模型实例化
    model = DomainAdaptationModel().to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()  # 分类损失
    domain_criterion = DomainAdaptationModule()  # 域适应性模块

    epochs = 150
    best_accuracy = 0
    for epoch in range(epochs):
        # 训练模型
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, domain_criterion, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # 测试
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        predictions = predict(model, x_test_tensor, device)
        # print(f"Test Predictions: {predictions.cpu().numpy()}")
        total_correct = 0
        total_samples = 0
        # print(y_test.shape)  # torch.Size([90])
        y_test = y_test.to(device)
        if epoch == 90:
            print(y_test)
            print(predictions)
        total_correct += (predictions == y_test).sum().item()
        total_samples += y_test.size(0)
        accuracy = total_correct / total_samples
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

    print(f'best_test_accuracy: {best_accuracy * 100:.2f}%')
    best_accuracy_list.append(best_accuracy * 100)

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 15})

    tick_labels = ['LH', 'RH', 'BH', 'BF']

    # 计算混淆矩阵
    y_test_cpu = y_test.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    confusion_mat = confusion_matrix(y_test_cpu, predictions_cpu)

    # 归一化混淆矩阵
    normalized_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    # 将数值格式化为百分比
    percentage_mat = np.around(normalized_mat, decimals=2)

    # 绘制混淆矩阵图形
    plt.figure(figsize=(8, 6))
    sns.heatmap(percentage_mat, annot=True, cmap='Blues', fmt='.2f', cbar=False)

    # 设置 x 轴和 y 轴刻度标签
    # plt.xticks(np.arange(len(tick_labels)), tick_labels)
    # plt.yticks(np.arange(len(tick_labels)), tick_labels)
    plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, ha='center')
    plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, va='center')

    # 设置图形的标题、轴标签和颜色栏
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    plt.close()

print("测试集最高准确率列表:")
print(best_accuracy_list)

# 计算总和
total = sum(best_accuracy_list)
# 计算平均值
average = total / len(best_accuracy_list)
# 打印平均值
print("测试集平均准确率为:", average)