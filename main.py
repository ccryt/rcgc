import argparse
import time
from functions import *
from tqdm import tqdm
from torch import optim
from model import *
from layers import *
import torch.nn.functional as F


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--gnnlayers', type=int, default=5, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128, help='hidden_num')
parser.add_argument('--dims', type=int, default=800, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.5, help='Loss balance parameter')
parser.add_argument('--beta', type=float, default=0.5, help='Loss balance parameter')
parser.add_argument('--threshold', type=float, default=0.95, help='the threshold')
parser.add_argument('--cluster_num', type=int, default=7, help='number of clusters.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


if args.dataset == 'cora':
    args.cluster_num = 7
    args.gnnlayers = 5
    args.dims = 800
    args.lr = 1e-5
elif args.dataset == 'citeseer':
    args.cluster_num = 6
    args.gnnlayers = 7
    args.dims = 1500
    args.lr = 1e-3


# two-sided hinge-quadratic =====
delta = 0.9  # 固定目标中心
tau_param = torch.nn.Parameter(torch.tensor(0.05))  # 可学习带宽
tau_optimizer = torch.optim.Adam([tau_param], lr=1e-3)
lambda_perturb = 0.5  # 带宽约束权重

acc_list, nmi_list, ari_list, f1_list = [], [], [], []

for seed in range(1):
    setup_seed(seed)

    features, true_labels, adj, _, _ = load_data(args.dataset)
    adj_tensor = adj.clone().detach().to(torch.float32)

    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    sm_fea_s = torch.FloatTensor(sm_fea_s)

    # 可逆网络
    reversible_net = reversible_model([features.shape[1]])
    optimizer_reversible_net = optim.SGD(reversible_net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)

    # 编码器网络
    model = Encoder_Net([features.shape[1]] + [args.dims], args.cluster_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()
        reversible_net.cuda()
        sm_fea_s = sm_fea_s.cuda()
        tau_param.data = tau_param.data.to(sm_fea_s.device)

    # 初始化
    best_acc, best_nmi, best_ari, best_f1, _, _, _ = clustering(sm_fea_s, true_labels, args.cluster_num)

    print('Start Training...')
    for epoch in tqdm(range(args.epochs)):
        optimizer_reversible_net.zero_grad()
        optimizer.zero_grad()
        tau_optimizer.zero_grad()

        # 生成增强特征
        aug_feature = reversible_net(sm_fea_s, True)

        # 获取嵌入
        z1, logits_z1 = model(sm_fea_s)
        z2, logits_z2 = model(aug_feature)

        # 可逆映射嵌入
        z11 = reversible_net(z1, True)
        z22 = reversible_net(z2, False)

        # 对比损失
        loss_1 = loss_cal(z1, z22)
        loss_2 = loss_cal(z2, z11)
        contra_loss = loss_1 + loss_2

        # 语义一致性损失
        cos_orig_rec = F.cosine_similarity(z1, z11, dim=1)
        cos_orig_pert = F.cosine_similarity(z1, z22, dim=1)
        mse_sem = F.mse_loss(cos_orig_rec, cos_orig_pert)

        # === Two-sided hinge-quadratic 正则 ===
        tau = F.softplus(tau_param)
        cos_sim = F.cosine_similarity(sm_fea_s, aug_feature, dim=1)
        lower = delta - tau
        upper = delta + tau
        loss_band = torch.clamp(cos_sim - upper, min=0).pow(2) + torch.clamp(lower - cos_sim, min=0).pow(2)
        loss_perturb = loss_band.mean()

        loss_semantic = mse_sem + lambda_perturb * loss_perturb

        # 伪标签监督阶段
        if epoch > 200:
            pseudo_z1 = torch.softmax(logits_z1, dim=-1)
            pseudo_z2 = torch.softmax(logits_z2, dim=-1)
            z_cluster = (z1 + z2) / 2

            p1 = target_distribution(pseudo_z1)
            p2 = target_distribution(pseudo_z2)
            kl_loss = F.kl_div(pseudo_z1.log(), p1, reduction='batchmean') + F.kl_div(pseudo_z2.log(), p2, reduction='batchmean')

            _, _, _, _, predict_labels, centers, dis = clustering(z_cluster, true_labels, args.cluster_num)
            high_confidence = torch.min(dis, dim=1).values.cpu()
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
            h_i = high_confidence_idx.numpy()
            y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]

            loss_match = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean() + (F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()

            total_loss = contra_loss + args.alpha * loss_semantic + args.beta * loss_match + args.beta * kl_loss
        else:
            total_loss = contra_loss + args.alpha * loss_semantic

        total_loss.backward()
        optimizer_reversible_net.step()
        optimizer.step()
        tau_optimizer.step()

        # 测试阶段
        if epoch % 5 == 0:
            model.eval()
            z1, _ = model(sm_fea_s)
            z2, _ = model(aug_feature)
            hidden_emb = (z1 + z2) / 2
            acc, nmi, ari, f1, aaa, _, _ = clustering(hidden_emb, true_labels, args.cluster_num)
            if acc >= best_acc:
                best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1


    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

    tqdm.write("Optimization Finished!")
    tqdm.write(f'best_acc: {best_acc}, best_nmi: {best_nmi}, best_ari: {best_ari}, best_f1: {best_f1}')

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
print(acc_list.mean())
print(nmi_list.mean())
print(ari_list.mean())
print(f1_list.mean())
