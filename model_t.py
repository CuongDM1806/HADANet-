import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import seaborn as sns
import os
import math

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
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(self.bn1(x1))
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x)
        x2 = F.relu(self.bn2(x2))
        x2 = self.dropout(x2)
        
        x = x2 * x1
        x = x.view(x.size(0), -1)
        return x

# 定义MLP分类器
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + 1e-8)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # 返回多个核的加权和

    def forward(self, source, target):
        # 特征归一化
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        batch_size = int(source.size()[0])
        if batch_size < 2:
            return torch.tensor(0.0).to(source.device)
        kernels = self.guassian_kernel(source, target)
        K_ss = kernels[:batch_size, :batch_size]
        K_tt = kernels[batch_size:, batch_size:]
        K_st = kernels[:batch_size, batch_size:]

        # 计算非对角线元素的和
        off_diag_ss = K_ss.sum() - torch.trace(K_ss)
        off_diag_tt = K_tt.sum() - torch.trace(K_tt)
        cross_sum = K_st.sum()

        # 无偏估计
        mmd = off_diag_ss / (batch_size * (batch_size - 1) + 1e-8) \
              + off_diag_tt / (batch_size * (batch_size - 1) + 1e-8) \
              - 2 * cross_sum / (batch_size * batch_size)
        return mmd
#--------------------------------------------------------



#--------------------------------------------------------

class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x)

class FeatureAligner(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(FeatureAligner, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(input_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x

class HybridDomainAdaptation(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(HybridDomainAdaptation, self).__init__()
        self.feature_aligner = FeatureAligner(input_size, hidden_size)
        self.domain_classifier = DomainClassifier(input_size, hidden_size)
        self.mmd_loss = MMDLoss()
        
    def forward(self, source_features, target_features, alpha=1.0):
        # 特征对齐
        aligned_source = self.feature_aligner(source_features)
        aligned_target = self.feature_aligner(target_features)
        
        # 域分类
        source_domain = self.domain_classifier(aligned_source)
        target_domain = self.domain_classifier(aligned_target)
        
        # 计算MMD损失
        mmd_loss = self.mmd_loss(aligned_source, aligned_target)
        
        # 计算域分类损失
        source_labels = torch.ones(source_domain.size(0), 1).to(source_domain.device)
        target_labels = torch.zeros(target_domain.size(0), 1).to(target_domain.device)
        domain_labels = torch.cat([source_labels, target_labels], dim=0)
        domain_outputs = torch.cat([source_domain, target_domain], dim=0)
        domain_loss = F.binary_cross_entropy(domain_outputs, domain_labels)
        
        # 梯度反转
        if self.training:
            domain_loss = -alpha * domain_loss
            
        return aligned_source, aligned_target, mmd_loss, domain_loss

# 定义完整的模型（包含特征提取、注意力机制、域适应、分类器）
class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = H_CNN()
        self.domain_adaptation = HybridDomainAdaptation(64 * 5 * 4)
        self.mlp = MLP(64 * 5 * 4, 512, 4)

    def forward(self, source_data, target_data, alpha=1.0):
        # 特征提取
        source_features = self.feature_extractor(source_data)
        target_features = self.feature_extractor(target_data)
        
        # 域适应
        aligned_source, aligned_target, mmd_loss, domain_loss = self.domain_adaptation(
            source_features, target_features, alpha
        )
        
        # 分类
        source_classification = self.mlp(aligned_source)
        target_classification = self.mlp(aligned_target)
        
        return source_classification, target_classification, mmd_loss, domain_loss


# 训练函数
def train(model, train_loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 动态调整域适应权重
    alpha = 2.0 / (1.0 + math.exp(-10 * epoch / epochs)) - 1
    
    for source_data, target_data, source_labels in train_loader:
        source_data, target_data, source_labels = source_data.to(device), target_data.to(device), source_labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        source_classification, target_classification, mmd_loss, domain_loss = model(source_data, target_data, alpha)
        
        # 分类损失
        source_labels = source_labels.type(torch.long)
        classification_loss = criterion(source_classification, source_labels)
        print(f"classification_loss:{classification_loss},mmd_loss: {mmd_loss}，domain_loss:{domain_loss}")
        # 总损失
        total_loss = classification_loss + 0.1 * mmd_loss + 0.01 * domain_loss
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 计算准确率
        _, predicted = torch.max(source_classification, 1)
        correct += (predicted == source_labels).sum().item()
        total += source_labels.size(0)
        
    accuracy = 100 * correct / total
    return total_loss.item() / len(train_loader), accuracy


# 预测函数
def predict(model, test_data, device):
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        features = model.feature_extractor(test_data)
        aligned_features, _, _, _ = model.domain_adaptation(features, features)
        output = model.mlp(aligned_features)
        _, predicted = torch.max(output, 1)
        return predicted


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_accuracy_list = []
for num in range(29, 30):
    # 数据加载
    feature_folder = "../physionet_feature_data/DE/"
    label_folder = "../physionet_feature_data/label/"
    
    features = []
    labels = []
    for i in range(num, num+1):
        feature_file = np.load(feature_folder + f"S{i}.npy")
        label_file = np.load(label_folder + f"S{i}.npy")
        features.append(feature_file)
        labels.append(label_file)
    
    features = np.array(features)
    labels = np.array(labels)
    
    # 数据预处理
    features = features.reshape(-1, 90, 4, 64, 5)
    features = np.transpose(features, (0, 1, 3, 4, 2))
    features = features.reshape((-1, 64, 5, 4))
    labels = labels.reshape((-1, 90, 4, 1)).mean(axis=2)
    labels = labels.reshape(-1)
    
    # 转换为tensor
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # 数据集划分
    indices = torch.arange(features.size(0))
    train_indices = []
    test_indices = []
    
    for i in range(0, len(indices), 6):
        for j in range(i, min(i + 5, len(indices))):
            train_indices.append(indices[j])
        if i + 5 < len(indices):
            test_indices.append(indices[i + 5])
    
    train_indices = torch.tensor(train_indices)
    test_indices = torch.tensor(test_indices)
    
    features_train = features[train_indices]
    features_test = features[test_indices]
    labels_train = labels[train_indices]
    labels_test = labels[test_indices]
    
    # 创建数据加载器
    train_dataset = TensorDataset(features_train, features_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
    
    # 模型实例化
    model = DomainAdaptationModel().to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    epochs = 200
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device, epoch, epochs)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # 测试
        features_test = features_test.to(device)
        predictions = predict(model, features_test, device)
        
        labels_test = labels_test.to(device)
        total_correct = (predictions == labels_test).sum().item()
        total_samples = labels_test.size(0)
        accuracy = total_correct / total_samples
        
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            print(f"New best accuracy: {best_accuracy * 100:.2f}%")
        
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        
        # 计算评估指标
        labels_test_cpu = labels_test.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()
        
        recall = recall_score(labels_test_cpu, predictions_cpu, average='macro')
        f1 = f1_score(labels_test_cpu, predictions_cpu, average='macro')
        print(f"Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    print(f'Best Test Accuracy: {best_accuracy * 100:.2f}%')
    
    # 保存最佳模型
    # if best_model_state is not None:
    #     folder_path = "D:/pythonProject/svr/wulaoshi"
    #     file_name = f"physionet_{num}_brain_model.pth"
    #     file_path = os.path.join(folder_path, file_name)
    #     torch.save(best_model_state, file_path)
    #
    best_accuracy_list.append(best_accuracy * 100)

print("测试集最高准确率列表:")
print(best_accuracy_list)

# 计算平均值
average = sum(best_accuracy_list) / len(best_accuracy_list)
print(f"测试集平均准确率为: {average:.2f}%")