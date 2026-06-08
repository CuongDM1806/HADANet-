# import torch
# import torch.nn.functional as F
#
#
# # 最大均值差异（MMD）损失
# def compute_mmd(source_features, target_features):
#     """
#     计算源域和目标域之间的最大均值差异（MMD）。
#     :param source_features: 源域特征，形状为 (batch_size, feature_dim)
#     :param target_features: 目标域特征，形状为 (batch_size, feature_dim)
#     :return: MMD损失值
#     """
#     # 计算源域和目标域特征的内积
#     xx = torch.mm(source_features, source_features.t())
#     yy = torch.mm(target_features, target_features.t())
#     xy = torch.mm(source_features, target_features.t())
#
#     # 核函数: RBF核（高斯核）
#     def rbf_kernel(x, y, gamma=1.0):
#         dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1, p=2)
#         return torch.exp(-gamma * dist)
#
#     # 计算RBF核矩阵
#     K_xx = rbf_kernel(source_features, source_features)
#     K_yy = rbf_kernel(target_features, target_features)
#     K_xy = rbf_kernel(source_features, target_features)
#
#     # 计算MMD损失
#     loss = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
#     return loss
#
#
# # 假设源域和目标域的特征已经通过某些特征提取方法得到
# def domain_adaptation_module(source_features, target_features):
#     """
#     域适应模块，计算源域和目标域之间的MMD损失。
#     :param source_features: 源域特征
#     :param target_features: 目标域特征
#     :return: 适应后的目标域特征
#     """
#     # 先将源域和目标域特征从 (batch_size, 64, 32, 32) 展平为 (batch_size, 64*32*32)
#     source_features = source_features.view(source_features.size(0), -1)  # 展平为 (batch_size, feature_dim)
#     target_features = target_features.view(target_features.size(0), -1)  # 展平为 (batch_size, feature_dim)
#
#     # 计算源域和目标域之间的MMD损失
#     mmd_loss = compute_mmd(source_features, target_features)
#
#     # 这里我们不进行训练，只返回适应后的目标域特征和MMD损失
#     # 假设我们通过某种方式调整目标域特征使其更接近源域（比如，后续可以做平滑处理等）
#
#     # 这里简单返回目标域特征和计算的MMD损失
#     return target_features, mmd_loss
#
#
# # 示例：生成一些虚拟数据，假设源域和目标域的特征都是(8, 64, 32, 32)形状
# source_features = torch.randn(32, 20, 5, 4)  # 源域特征，8个样本，每个样本64个通道，32x32的特征图
# target_features = torch.randn(32, 20, 5, 4)  # 目标域特征，8个样本，每个样本64个通道，32x32的特征图
#
# # 使用域适应模块进行适应
# adapted_target_features, mmd_loss = domain_adaptation_module(source_features, target_features)
# adapted_target_features = adapted_target_features.reshape(32, 20, 5, 4)
#
# print("MMD Loss:", mmd_loss.item())
# print(adapted_target_features.shape)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
#
#
# # 定义卷积神经网络（CNN）特征提取模块
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 5 * 4, 256)
#         self.fc2 = nn.Linear(256, 128)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         # x = x.view(x.size(0), -1)  # Flatten the feature map
#         # x = F.relu(self.fc1(x))
#         # x = self.fc2(x)
#         # print(x.shape)
#         return x
#
#
# # 定义最大均值差异（MMD）损失函数
# def compute_mmd(source_features, target_features):
#     # 计算源域和目标域的均值
#     source_mean = source_features.mean(0)
#     target_mean = target_features.mean(0)
#
#     # 计算MMD损失：L2范数（欧氏距离）
#     mmd_loss = torch.norm(source_mean - target_mean, p=2)
#     return mmd_loss
#
#
# # 定义一个简单的域适应网络
# class DomainAdaptationNetwork(nn.Module):
#     def __init__(self):
#         super(DomainAdaptationNetwork, self).__init__()
#         self.feature_extractor = FeatureExtractor()
#
#     def forward(self, source_data, target_data):
#         source_features = self.feature_extractor(source_data)
#         target_features = self.feature_extractor(target_data)
#
#         # 计算MMD损失
#         mmd_loss = compute_mmd(source_features, target_features)
#
#         return source_features, target_features, mmd_loss
#
#
# # 创建模型
# model = DomainAdaptationNetwork()
#
# # 模拟源域和目标域的数据（假设为随机数据）
# source_data = torch.randn(8010, 64, 5, 4)  # 源域数据
# target_data = torch.randn(90, 64, 5, 4)  # 目标域数据
#
# # 优化器
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
# # 训练步骤（一个简单的示例）
# for epoch in range(10):
#     print(epoch)
#     model.train()
#     optimizer.zero_grad()
#
#     # 正向传播
#     features, test_features, mmd_loss = model(source_output, target_output)
#
#     # 假设源域和目标域有相同的标签，且我们没有真正的标签信息
#     # 可以加上分类损失（如交叉熵损失），但是这里我们只聚焦于MMD损失
#     total_loss = mmd_loss
#
#     # 反向传播
#     total_loss.backward()
#     optimizer.step()
#
# # 测试模型
# model.eval()
# with torch.no_grad():
#     source_features, target_features, mmd_loss = model(source_data, target_data)
#     print(f"Final MMD Loss: {mmd_loss.item():.4f}")
#     # print(source_data.shape)
#     print(source_features.shape)
#     print(target_features.shape)
import torch
import torch.nn as nn
import torch.optim as optim


# 定义特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        # 这里假设特征提取器是一个简单的卷积网络
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 5 * 4, output_dim)  # 假设最后的输出维度为output_dim

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x


# 定义域判别器（Domain Classifier）
class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # 二分类任务：源域 vs 目标域

    def forward(self, x):
        return self.fc(x)


# 定义整个域适应模块
class DomainAdaptationModule(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(DomainAdaptationModule, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, feature_dim)
        self.domain_classifier = DomainClassifier(feature_dim)

    def forward(self, source_data, target_data):
        # 提取源域和目标域特征
        source_features = self.feature_extractor(source_data)
        target_features = self.feature_extractor(target_data)

        # 在训练时，使用反向传播来优化域分类器（使得源域和目标域不可区分）
        source_domain_preds = self.domain_classifier(source_features)
        target_domain_preds = self.domain_classifier(target_features)

        return source_features, target_features, source_domain_preds, target_domain_preds


# 创建模型
input_dim = 64  # 输入特征的通道数
feature_dim = 1280  # 特征提取后的维度
model = DomainAdaptationModule(input_dim, feature_dim)

# 创建数据
source_data = torch.randn(8010, 64, 5, 4)  # 假设源域数据
target_data = torch.randn(90, 64, 5, 4)  # 假设目标域数据

# 训练时的优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()  # 用于计算域分类的损失


# 一个简单的训练步骤
def train_step(source_data, target_data):
    model.train()

    optimizer.zero_grad()

    source_features, target_features, source_domain_preds, target_domain_preds = model(source_data, target_data)

    # 计算源域的域分类损失
    source_labels = torch.zeros(source_data.size(0), dtype=torch.long)  # 0表示源域
    target_labels = torch.ones(target_data.size(0), dtype=torch.long)  # 1表示目标域

    loss_source = criterion(source_domain_preds, source_labels)
    loss_target = criterion(target_domain_preds, target_labels)

    # 总损失是源域和目标域的损失之和
    total_loss = loss_source + loss_target

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), source_features, target_features


# 训练一个batch的源域和目标域数据
loss, source_features, target_features = train_step(source_data, target_data)
print(f'Training loss: {loss}')
source_features = source_features.reshape(8010, 64, 5, 4)
target_features = target_features.reshape(90, 64, 5, 4)
print(source_features.shape)
print(target_features.shape)
