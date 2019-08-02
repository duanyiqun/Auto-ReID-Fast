import torch.nn as nn 
import torch

def cross_entropy():
    return nn.CrossEntropyLoss()


class TripletLoss(nn.Module):
    ####### One function to calculate online triplet loss ###########
    #  For Prof Tao
    #  triplet loss  的表达应该是如下， 论文原文中应该把正样本和负样本符号写反了 这里的实现进行了更新
    #  L_triplet = \sum ||F_i - F_p ||2 - ||F_i - F_n||2
    #  F_p 距离最大的负样本， F_n 是距离最小的负样本
    #  在工程化的时候往往 负样本不一定采用 最大的负样本，而是直接去采样 false alarm 图片，这样更robust并且 训练稳定一些。 
    #  在openset的时候往往正样本会采用一个平均feature （就是gallary 做比对的metric 作为anchor）
    #  如果严格的triplet应该要计算所有的三元组， 但是基于类别的采样可以减少计算量，放到一个batch 里面。
    #  为了计算加速，计算距离，不采取组计算，直接采用矩阵计算， 如下：

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数

    def forward(self, inputs, labels):

        n = inputs.size(0)  # 获取batch_size
        # Compute pairwise distance, replace by the official when merged

        ### 计算L2 距离 
        # 欧式距离|a-b|^2 = a^2 -2ab + b^2, 
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  
        # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维  i，j 
        dist = dist + dist.t()  # 这样对角线代表平方和
        dist.addmm_(1, -2, inputs, inputs.t())  # |a-b|^2 = a^2 -2ab + b^2
        dist = dist.clamp(min=1e-12).sqrt()  

        ### 计算完毕

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 只有相同label mask 才为1 不同label 为零，避免循环， 加速计算。
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss



class retrieval_loss(nn.Module):
    def __init__(self, lamb = 0.5):
        super().__init__()
        self.lamb = lamb
        self.cross_entropy = cross_entropy()
        self.triplet_loss = TripletLoss()
        
        
    def forward(self, inputs, labels):
        return self.lamb*self.cross_entropy(inputs, labels) + (1-self.lamb) * self.triplet_loss(inputs, labels)

    

def get_loss(args):
    return eval(args.loss_f)()