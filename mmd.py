import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import math
from collections import defaultdict
import platform

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 根据操作系统设置并行处理
if platform.system() == 'Windows':
    num_workers = 0  # Windows不支持fork，设置为0
else:
    num_workers = min(4, os.cpu_count())  # 使用最多4个工作进程


# 定义梯度反转层 (GRL)
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        return -alpha * grad_output, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversal.apply(x, alpha)


# 定义H-CNN 输入 [batch_size, in_channel, 5, 4]
class H_CNN(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(H_CNN, self).__init__()
        # 水平卷积
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(5, 1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 垂直卷积
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 4)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x1 = self.h_conv(x)
        x2 = self.v_conv(x)

        # 融合特征
        x = x1 * x2
        x = self.fusion(x)

        x = x.view(x.size(0), -1)
        return x


# 定义MLP分类器
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 改进的MMD损失
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # 更高效的距离计算
        total_square = torch.sum(total ** 2, dim=1, keepdim=True)
        dists = total_square + total_square.t() - 2 * torch.mm(total, total.t())
        dists = dists.clamp(min=0.0)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(dists.data) / (n_samples ** 2 - n_samples + 1e-8)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_val = 0
        for bandwidth_temp in bandwidth_list:
            kernel_val += torch.exp(-dists / bandwidth_temp)

        return kernel_val

    def forward(self, source, target):
        # 特征归一化
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        batch_size_s = source.size(0)
        batch_size_t = target.size(0)

        if batch_size_s < 2 or batch_size_t < 2:
            return torch.tensor(0.0, device=source.device)

        kernels = self.guassian_kernel(source, target)
        K_ss = kernels[:batch_size_s, :batch_size_s]
        K_tt = kernels[batch_size_s:, batch_size_s:]
        K_st = kernels[:batch_size_s, batch_size_s:]

        # 无偏估计
        mmd = (torch.sum(K_ss) - torch.trace(K_ss)) / (batch_size_s * (batch_size_s - 1)) \
              + (torch.sum(K_tt) - torch.trace(K_tt)) / (batch_size_t * (batch_size_t - 1)) \
              - 2 * torch.sum(K_st) / (batch_size_s * batch_size_t)

        return mmd


# 改进的域分类器
class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(DomainClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 改进的特征对齐器
class FeatureAligner(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(FeatureAligner, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.Tanh()  # 使用Tanh限制特征范围
        )

    def forward(self, x):
        return self.net(x)


# 改进的域适应模块
class HybridDomainAdaptation(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(HybridDomainAdaptation, self).__init__()
        self.feature_aligner = FeatureAligner(input_size, hidden_size)
        self.domain_classifier = DomainClassifier(input_size, hidden_size)
        self.mmd_loss = MMDLoss()
        self.grl = GRL()  # 梯度反转层

    def forward(self, source_features, target_features, alpha=1.0):
        # 特征对齐
        aligned_source = self.feature_aligner(source_features)
        aligned_target = self.feature_aligner(target_features)

        # 应用梯度反转
        grl_source = self.grl(aligned_source, alpha)
        grl_target = self.grl(aligned_target, alpha)

        # 域分类
        source_domain = self.domain_classifier(grl_source)
        target_domain = self.domain_classifier(grl_target)

        # 计算MMD损失
        mmd_loss = self.mmd_loss(aligned_source, aligned_target)

        # 计算域分类损失
        source_labels = torch.ones(source_domain.size(0), 1, device=source_domain.device)
        target_labels = torch.zeros(target_domain.size(0), 1, device=target_domain.device)
        domain_labels = torch.cat([source_labels, target_labels], dim=0)
        domain_outputs = torch.cat([source_domain, target_domain], dim=0)
        domain_loss = F.binary_cross_entropy(domain_outputs, domain_labels)

        return aligned_source, aligned_target, mmd_loss, domain_loss


# 定义完整的模型
class DomainAdaptationModel(nn.Module):
    def __init__(self, input_size=64 * 5 * 4):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = H_CNN()
        self.domain_adaptation = HybridDomainAdaptation(input_size)
        self.mlp = MLP(input_size, 512, 4)

        # 特征归一化
        self.feature_norm = nn.BatchNorm1d(input_size)

    def forward(self, source_data, target_data, alpha=1.0):
        # 特征提取
        source_features = self.feature_extractor(source_data)
        target_features = self.feature_extractor(target_data)

        # 特征归一化
        source_features = self.feature_norm(source_features)
        target_features = self.feature_norm(target_features)

        # 域适应
        aligned_source, aligned_target, mmd_loss, domain_loss = self.domain_adaptation(
            source_features, target_features, alpha
        )

        # 分类
        source_classification = self.mlp(aligned_source)
        target_classification = self.mlp(aligned_target)

        return source_classification, target_classification, mmd_loss, domain_loss


# 改进的训练函数
def train(model, train_loader, optimizer, criterion, device, epoch, epochs, history):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 动态调整域适应权重
    p = epoch / epochs
    alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1

    # 动态调整损失权重
    domain_weight = 0.1 * min(p * 3, 1.0)  # 逐步增加
    mmd_weight = 0.1 * max(1 - p * 2, 0.1)  # 逐步减少

    for batch_idx, (source_data, target_data, source_labels) in enumerate(train_loader):
        # 确保所有数据都在正确设备上
        source_data = source_data.to(device)
        target_data = target_data.to(device)
        source_labels = source_labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        source_classification, target_classification, mmd_loss, domain_loss = model(source_data, target_data, alpha)

        # 分类损失
        source_labels = source_labels.type(torch.long)
        classification_loss = criterion(source_classification, source_labels)

        # 总损失 - 关键修改：降低域适应损失权重
        total_loss_value = classification_loss + 0.01 * mmd_loss + 0.001 * domain_loss

        # 记录损失
        history['classification_loss'].append(classification_loss.item())
        history['mmd_loss'].append(mmd_loss.item() if not torch.isnan(mmd_loss) else 0)
        history['domain_loss'].append(domain_loss.item())
        history['total_loss'].append(total_loss_value.item())

        # 反向传播
        total_loss_value.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(source_classification, 1)
        correct += (predicted == source_labels).sum().item()
        total += source_labels.size(0)

        # 每10个batch打印一次
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)}")
            print(
                f"  Cls Loss: {classification_loss.item():.4f} | MMD Loss: {mmd_loss.item():.4f} | Domain Loss: {domain_loss.item():.4f}")
            print(f"  Total Loss: {total_loss_value.item():.4f} | Alpha: {alpha:.3f}")
            print(f"  Domain Weight: {domain_weight:.4f} | MMD Weight: {mmd_weight:.4f}")

    accuracy = 100 * correct / total
    return total_loss_value.item() / len(train_loader), accuracy


# 预测函数
def predict(model, test_data, device):
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        features = model.feature_extractor(test_data)
        features = model.feature_norm(features)
        aligned_features, _, _, _ = model.domain_adaptation(features, features)
        output = model.mlp(aligned_features)
        _, predicted = torch.max(output, 1)
        return predicted


# 设置随机种子确保可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

best_accuracy_list = []
for num in range(29, 30):
    print(f"\n\n===== Processing Subject {num} =====")

    # 数据加载
    feature_folder = "../physionet_feature_data/DE/"
    label_folder = "../physionet_feature_data/label/"

    features = []
    labels = []
    for i in range(num, num + 1):
        feature_file = np.load(os.path.join(feature_folder, f"S{i}.npy"))
        label_file = np.load(os.path.join(label_folder, f"S{i}.npy"))
        features.append(feature_file)
        labels.append(label_file)

    features = np.array(features)
    labels = np.array(labels)

    # 改进的数据预处理
    # 1. 添加通道维度
    features = features.reshape(-1, 90, 4, 64, 5)
    features = np.transpose(features, (0, 1, 4, 3, 2))  # [batch, trial, channel, height, width]
    features = features.reshape((-1, 64, 5, 4))

    # 2. 标签处理 - 使用众数而非平均值
    labels = labels.reshape((-1, 90, 4, 1))
    labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), 2, labels)
    labels = labels.reshape(-1)

    # 3. 数据标准化
    mean = features.mean(axis=(0, 2, 3), keepdims=True)
    std = features.std(axis=(0, 2, 3), keepdims=True)
    features = (features - mean) / (std + 1e-8)

    # 转换为tensor
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)  # 使用long类型

    # 检查标签分布
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")

    # 数据集划分
    indices = torch.arange(features.size(0))
    train_indices = []
    test_indices = []

    for i in range(0, len(indices), 6):
        for j in range(i, min(i + 4, len(indices))):# 4个训练样本
            train_indices.append(indices[j])
        if i + 5 < len(indices):
            test_indices.append(indices[i + 4])  # 1个验证样本
        test_indices.append(indices[i + 5])  # 1个测试样本

        train_indices = torch.tensor(train_indices)
        test_indices = torch.tensor(test_indices)

        features_train = features[train_indices]
        features_test = features[test_indices]
        labels_train = labels[train_indices]
        labels_test = labels[test_indices]

        # 创建数据加载器
        train_dataset = TensorDataset(features_train, features_train, labels_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

        # 模型实例化并移动到设备
        model = DomainAdaptationModel().to(device)

        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

        # 使用CosineAnnealingLR和ReduceLROnPlateau组合
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        criterion = nn.CrossEntropyLoss()

        # 训练循环
        epochs = 150
        best_accuracy = 0
        best_model_state = None
        no_improve = 0
        patience = 15
        history = defaultdict(list)

        for epoch in range(epochs):
            train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device, epoch, epochs,
                                               history)
        print(f"\nEpoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # 测试
        features_test = features_test.to(device)
        predictions = predict(model, features_test, device)

        labels_test = labels_test.to(device)
        total_correct = (predictions == labels_test).sum().item()
        total_samples = labels_test.size(0)
        accuracy = total_correct / total_samples

        # 更新学习率
        scheduler1.step()
        scheduler2.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            no_improve = 0
            print(f"New best accuracy: {best_accuracy * 100:.2f}%")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs")

        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        # 计算评估指标
        labels_test_cpu = labels_test.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()

        recall = recall_score(labels_test_cpu, predictions_cpu, average='macro')
        f1 = f1_score(labels_test_cpu, predictions_cpu, average='macro')
        print(f"Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # 早停机制
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
        break

    print(f'\nSubject {num} Best Test Accuracy: {best_accuracy * 100:.2f}%')

    # 保存最佳模型
    if best_model_state is not None:
        folder_path = "models"
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"physionet_{num}_best_model.pth"
        file_path = os.path.join(folder_path, file_name)
        torch.save(best_model_state, file_path)
        print(f"Saved best model for subject {num}")

    best_accuracy_list.append(best_accuracy * 100)

print("\n\n===== Final Results =====")
print("测试集最高准确率列表:")
print(best_accuracy_list)

# 计算平均值
average = sum(best_accuracy_list) / len(best_accuracy_list)
print(f"测试集平均准确率为: {average:.2f}%")

# 保存准确率结果
with open("accuracy_results.txt", "w") as f:
    f.write(f"Subject Accuracies: {best_accuracy_list}\n")
    f.write(f"Average Accuracy: {average:.2f}%\n")