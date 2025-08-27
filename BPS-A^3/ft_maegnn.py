import torch as th
import torch.nn.functional as F
import pandas as pd
import dgl
import torch.utils.data as Data
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Precision, Recall, Loss
from sklearn.metrics import accuracy_score
import os
import shutil
import argparse
import sys
import logging
from torch.optim import lr_scheduler
from models.combine_model import triatten
import networkx as nx
from torchvision import datasets, transforms
import warnings

warnings.filterwarnings("ignore",
                        message="The number of unique classes is greater than 50%",
                        category=UserWarning)

th.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mirage', choices=['ciciot','andmal', 'cstnet','vpnapp','vpnser','mirage'])
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--m', type=float, default=0.4, help='the factor balancing Transformer and DiESAGE prediction')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--pretrained_ckpt', default=True)
parser.add_argument('--checkpoint_dir', default=None)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--sage_lr', type=float, default=1e-3)
parser.add_argument('--mae_lr', type=float, default=5e-5)

args = parser.parse_args()
batch_size = args.batch_size
m = args.m
epochs = args.epochs
pretrained_ckpt = args.pretrained_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
dropout = args.dropout
sage_lr = args.sage_lr
mae_lr = args.mae_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}/gnn'.format(dataset)
    # ckpt_dir = './checkpoint/{}_{}'.format('GNN', dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='a')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))


# === Snake Zigzag 索引构造 ===
def get_snake_index_2d(height=8, width=40):
    index = th.zeros((height, width), dtype=th.long)
    counter = 0
    for col in range(width):
        if col % 2 == 0:
            for row in range(height):
                index[row, col] = counter
                counter += 1
        else:
            for row in reversed(range(height)):
                index[row, col] = counter
                counter += 1
    return index.view(-1)  # shape [320]

# === 字节和方向处理函数（可选 Zigzag 和方向翻转） ===
def preprocess_bytes_and_dirs(bytes_list, dir_list, snake_index, normalize,
                              apply_zigzag=True, apply_direction=True):
    packets = []
    for i in range(5):
        pkt = th.FloatTensor(bytes_list[i])  # [320]

        if apply_zigzag:
            pkt = pkt[snake_index]

        pkt = pkt.view(8, 40)

        if apply_direction and dir_list[i] != 1:
            pkt = th.flip(pkt, dims=[-1])  # 左右翻转

        packets.append(pkt)

    img = th.cat(packets, dim=0).unsqueeze(0)  # [1, 40, 40]
    img = img / 255.0
    img = normalize(img)
    return img  # [1, 40, 40]

# === 加载数据并预处理 ===
flows = pd.read_csv('../datasets/' + dataset + '.csv')
flows = flows.sample(frac=1, random_state=123)
num_class = flows['label'].nunique()

# 字段转换与组合
flows['sip'] = flows['sip'].astype(str)
flows['dip'] = flows['dip'].astype(str)
flows['sport'] = flows['sport'].astype(str)
flows['dport'] = flows['dport'].astype(str)
flows['src_node'] = flows['sip'] + ':' + flows['sport']
flows['dst_node'] = flows['dip'] + ':' + flows['dport']
flows['bytes'] = flows['bytes'].apply(eval)
flows['dir_list'] = flows['direction_list'].apply(eval)

# Snake索引与归一化器
snake_index = get_snake_index_2d()
normalize = transforms.Normalize(mean=[0.5], std=[0.5])

# 替换 flows['bytes']：将其直接变为 [1, 40, 40] Tensor
flows['bytes'] = [
    preprocess_bytes_and_dirs(b, d, snake_index, normalize,
                               apply_zigzag=True,
                               apply_direction=True)
    for b, d in zip(flows['bytes'], flows['dir_list'])
]

# === 构造划分 Mask ===
num_edges = flows.shape[0]
train_index = range(int(num_edges * 0.8))
valid_index = range(int(num_edges * 0.8), int(num_edges * 0.9))
test_index = range(int(num_edges * 0.9), num_edges)

train_mask = th.zeros(num_edges, dtype=th.bool)
valid_mask = th.zeros(num_edges, dtype=th.bool)
test_mask = th.zeros(num_edges, dtype=th.bool)
train_mask[train_index] = True
valid_mask[valid_index] = True
test_mask[test_index] = True

flows['train'] = train_mask
flows['valid'] = valid_mask
flows['test'] = test_mask

# === 构建 DGL 图 ===
G = nx.from_pandas_edgelist(
    flows,
    source='dst_node',
    target='src_node',
    edge_attr=['label', 'bytes', 'train', 'valid', 'test'],
    create_using=nx.MultiDiGraph()
)
G = G.to_directed()
G = dgl.from_networkx(G, edge_attrs=['label', 'bytes', 'train', 'valid', 'test'])

# === 初始化特征字段 ===
model = triatten(num_class=num_class, m=m)  # ← 请确保 triatten 已定义
G.edata['cls_feat'] = th.zeros(G.num_edges(), model.feat_dim)
G.ndata['node_feat'] = th.ones(G.num_nodes(), model.feat_dim)

# === 加载预训练 BERT（可选）===
# if pretrained_ckpt is not None:
#     ckpt = th.load(pretrained_ckpt, map_location=gpu)
#     model.pre_model.load_state_dict(ckpt['pre_model'], strict=True)
if pretrained_ckpt:
    checkpoint_path = f'checkpoint/{dataset}/normal/checkpoint.pth'
    ckpt = th.load(checkpoint_path, map_location=gpu)
    model.pre_model.load_state_dict(ckpt['pre_model'], strict=True)


# create index loader
train_idx = Data.TensorDataset(th.LongTensor(train_index))
val_idx = Data.TensorDataset(th.LongTensor(valid_index))
test_idx = Data.TensorDataset(th.LongTensor(test_index))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)


# model.load_state_dict(th.load('output_dir/pretrained-model.pth', map_location=gpu)['model'], strict=False)


# Training
def update_feature():
    global model, G
    # no gradient needed, uses a large batch-size to speed up the process
    dataloader = Data.DataLoader(Data.TensorDataset(G.edata['bytes']), batch_size=1024)
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            (payloads,) = [x.to(gpu) for x in batch]
            output = model.pre_model.forward_features(payloads)
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    G = G.to(cpu)
    G.edata['cls_feat'] = cls_feat
    return G


optimizer = th.optim.Adam([
        {'params': model.pre_model.parameters(), 'lr': mae_lr},
        # {'params': model.classifier.parameters(), 'lr': mae_lr},
        {'params': model.gnn.parameters(), 'lr': sage_lr},
    ], lr=1e-3
)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, G, optimizer
    model.train()
    model = model.to(gpu)
    G = G.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = G.edata['train'][idx]
    y_pred = model(G, idx)[train_mask]
    y_true = G.edata['label'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    G.edata['cls_feat'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    # scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, G
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        G = G.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(G, idx)
        y_true = G.edata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)

recall = Recall(average='macro')
precision = Precision(average='macro')
F1 = (precision * recall * 2 / (precision + recall))

metrics={
    'acc': Accuracy(),
    'recall': recall,
    'precision': precision,
    'F1': F1,
    'nll': Loss(th.nn.CrossEntropyLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_recall, test_precision, test_F1, test_nll = metrics["acc"],  metrics["recall"], metrics["precision"], metrics["F1"], metrics["nll"]
    logger.info(
        "\rEpoch: {}  Train acc: {:.4f}  Val acc: {:.4f}  Test acc: {:.4f}  recall: {:.4f}  precision: {:.4f}  F1_score: {:.4f}"
        .format(trainer.state.epoch, train_acc, val_acc, test_acc, test_recall, test_precision, test_F1)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'pre_model': model.pre_model.state_dict(),
                'gnn': model.gnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
G = update_feature()
trainer.run(idx_loader, max_epochs=epochs)
