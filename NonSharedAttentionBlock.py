# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         # 全局平均池化层
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             # 第一个全连接层，通道数缩小
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             # 第二个全连接层，恢复通道数
#             nn.Linear(in_channels // reduction, in_channels, bias=False),
#             # Sigmoid激活函数，输出注意力权重
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         print("进入到：ChannelAttention")
#         batch_size, channels, _, _ = x.size()
#         # 对输入进行全局平均池化，得到每个通道的全局信息
#         avg_out = self.avg_pool(x).view(batch_size, channels)
#         # 通过全连接层计算通道注意力权重
#         channel_weights = self.fc(avg_out).view(batch_size, channels, 1, 1)
#         # 返回加权后的输入
#         return x * channel_weights
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         # 使用一个7x7的卷积核来生成空间注意力图
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         # Sigmoid激活函数，得到注意力权重
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         print("进入到：SpatialAttention")
#         # 分别计算输入的平均值和最大值
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         # 将平均池化和最大池化的结果拼接
#         spatial_features = torch.cat([avg_out, max_out], dim=1)
#         # 通过卷积计算空间注意力图
#         spatial_attention_map = self.conv1(spatial_features)
#         # 返回加权后的输入
#         return x * self.sigmoid(spatial_attention_map)
#
#
# class NonSharedAttentionBlock1(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(NonSharedAttentionBlock1, self).__init__()
#         # 初始化通道注意力模块
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         # 初始化空间注意力模块
#         self.spatial_attention = SpatialAttention()
#
#     def forward(self, x):
#         # 先应用通道注意力
#         x = self.channel_attention(x)
#         return x
#
# class NonSharedAttentionBlock2(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(NonSharedAttentionBlock2, self).__init__()
#         # 初始化通道注意力模块
#         self.channel_attention = ChannelAttention(in_channels, reduction)
#         # 初始化空间注意力模块
#         self.spatial_attention = SpatialAttention()
#
#     def forward(self, x):
#         # 然后应用空间注意力
#         x = self.spatial_attention(x)
#         return x
#
#
# class DomainSpecificAttentionNet(nn.Module):
#     def __init__(self, in_channels, num_classes, reduction=16):
#         super(DomainSpecificAttentionNet, self).__init__()
#         # 源域的注意力模块
#         self.source_attention_block = NonSharedAttentionBlock1(in_channels, reduction)
#         # 目标域的注意力模块
#         self.target_attention_block = NonSharedAttentionBlock2(in_channels, reduction)
#
#         # 分类的共享卷积层
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#         # 全连接层，输出类别数
#         self.fc = nn.Linear(in_channels * 5 * 4, num_classes)  # 假设输入尺寸为5x4
#
#     def forward(self, source, target, channel):
#         # 对源域输入应用注意力模块
#         source_features = self.source_attention_block(source)
#         print(source_features.shape)
#
#         # 对目标域输入应用注意力模块
#         target_features = self.target_attention_block(target)
#         print(target_features.shape)
#
#         # 通过注意力聚焦相关特征
#         # 展开到 [32, 64, 1024] 的形状
#         F1_flatten = source_features.view(90, channel, -1)
#         F2_flatten = target_features.view(90, channel, -1)
#
#         # 计算注意力权重（可以使用点积或者其他方法）
#         attention_weights = torch.matmul(F1_flatten, F2_flatten.transpose(1, 2))
#
#         # 对注意力矩阵进行 softmax 归一化
#         attention_weights = F.softmax(attention_weights, dim=-1)
#
#         # 对 F2 的特征进行加权
#         weighted_F2 = torch.matmul(attention_weights, F2_flatten)
#
#         # 将加权后的特征图恢复原形状
#         output = weighted_F2.view(90, channel, 5, 4)
#
#         return output
#
#
# # 示例用法
# if __name__ == "__main__":
#     # 假设输入：批量大小=90，通道数=64，尺寸=5x4
#     source_input = torch.randn(8010, 64, 5, 4)
#     print(source_input.shape)
#     # 从 source_input 中选择前 90 个样本
#     source_input = source_input[::89]  # 形状将变为 (90, 64, 5, 4)
#     print(source_input.shape)
#     target_input = torch.randn(90, 64, 5, 4)
#
#     model = DomainSpecificAttentionNet(in_channels=64, num_classes=4)
#     output = model(source_input, target_input, 64)
#
#     print(output.shape)  # 预期输出：torch.Size([90, 64, 5, 4])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ChannelAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ChannelAttentionBlock, self).__init__()
#         # MLP层
#         self.fc1 = nn.Linear(in_channels, in_channels // 16)
#         self.fc2 = nn.Linear(in_channels // 16, in_channels)
#
#     def forward(self, x):
#         # 全局平均池化
#         avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
#         max_pool, _ = torch.max(x, dim=-1, keepdim=True)
#         max_pool, _ = torch.max(max_pool, dim=-2, keepdim=True)
#
#         # 通过MLP生成注意力权重
#         avg_pool = avg_pool.view(avg_pool.size(0), -1)
#         max_pool = max_pool.view(max_pool.size(0), -1)
#         x_emb = avg_pool + max_pool
#         x_emb = F.relu(self.fc1(x_emb))
#         x_emb = torch.sigmoid(self.fc2(x_emb))
#
#         # 扩展成相同维度并重标定
#         x_emb = x_emb.view(x_emb.size(0), -1, 1, 1)
#         return x * x_emb
#
#
# class SpatialAttentionBlock(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(SpatialAttentionBlock, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 通道压缩
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         max_pool, _ = torch.max(x, dim=1, keepdim=True)
#
#         # 拼接后通过卷积
#         spatial_attention_map = torch.cat([avg_pool, max_pool], dim=1)
#         spatial_attention_map = self.conv(spatial_attention_map)
#         spatial_attention_map = self.sigmoid(spatial_attention_map)
#
#         # 重标定
#         return x * spatial_attention_map
#
# class AttentionModule(nn.Module):
#     def __init__(self, in_channels):
#         super(AttentionModule, self).__init__()
#         self.channel_attention = ChannelAttentionBlock(in_channels)
#         self.spatial_attention = SpatialAttentionBlock(kernel_size=3)
#
#     def forward(self, x):
#         x = self.channel_attention(x)  # 通道注意力
#         x = self.spatial_attention(x)  # 空间注意力
#         return x
#
#
# # 用于源域的注意力模块
# class SourceDomainAttentionNetwork(nn.Module):
#     def __init__(self):
#         super(SourceDomainAttentionNetwork, self).__init__()
#         self.attention_block = AttentionModule(in_channels=64)  # 64个通道
#         self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.attention_block(x)
#         x = self.conv(x)  # 保持输出形状不变
#         return x
#
#
# # 用于目标域的注意力模块
# class TargetDomainAttentionNetwork(nn.Module):
#     def __init__(self):
#         super(TargetDomainAttentionNetwork, self).__init__()
#         self.attention_block = AttentionModule(in_channels=64)  # 64个通道
#         self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.attention_block(x)
#         x = self.conv(x)  # 保持输出形状不变
#         return x
#
#
# # 网络实例
# source_domain_net = SourceDomainAttentionNetwork()
# target_domain_net = TargetDomainAttentionNetwork()
#
# # 示例输入
# source_input = torch.randn(8010, 64, 5, 4)
# target_input = torch.randn(90, 64, 5, 4)
#
# source_output = source_domain_net(source_input)
# target_output = target_domain_net(target_input)
#
# print(source_output.shape)  # 应该是 (8010, 64, 5, 4)
# print(target_output.shape)  # 应该是 (90, 64, 5, 4)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 使用全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用全连接层减少通道数
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入进行全局平均池化
        avg_pool = self.global_avg_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)  # Flatten for FC layers
        avg_pool = self.fc1(avg_pool)
        avg_pool = F.relu(avg_pool)
        avg_pool = self.fc2(avg_pool)
        avg_pool = self.sigmoid(avg_pool).view(avg_pool.size(0), -1, 1, 1)
        # 乘以输入特征图进行通道加权
        return x * avg_pool.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用卷积层生成空间注意力图
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对每个通道的空间维度进行卷积，生成空间注意力图
        spatial_attention_map = self.conv(x)
        spatial_attention_map = self.sigmoid(spatial_attention_map)
        # 乘以输入特征图进行空间加权
        return x * spatial_attention_map.expand_as(x)


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(DualAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels, kernel_size)

    def forward(self, x):
        # 先通过通道注意力，再通过空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# 测试双共享注意力模块
if __name__ == '__main__':
    # 输入特征 shape = (32, 8, 5, 4)
    x = torch.randn(32, 8, 5, 4)

    # 创建并运行注意力模块
    dam = DualAttentionModule(in_channels=8)
    output = dam(x)

    print(f'输入特征形状: {x.shape}')
    print(f'输出特征形状: {output.shape}')
