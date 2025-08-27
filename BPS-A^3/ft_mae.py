import torch as th
import torch.nn.functional as F
import sys
import torch.utils.data as Data
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Precision, Recall, Fbeta, Loss
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import argparse, shutil, logging
from torchvision import datasets, transforms
from models.models_2D import TrafficTransformer
from functools import partial
import torch.nn as nn
import ast
import warnings

warnings.filterwarnings("ignore",
                        message="The number of unique classes is greater than 50%",
                        category=UserWarning)

th.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=160)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--mae_lr', type=float, default=2e-4)
parser.add_argument('--dataset', default='mirage', choices=['ciciot','andmal', 'cstnet','vpnapp','vpnser','mirage'])
parser.add_argument('--checkpoint_dir', default=None)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
mae_lr = args.mae_lr
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir

pretrained_model_path = 'output_dir/pretrained-model.pth'

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}/normal'.format(dataset)
else:
    ckpt_dir = checkpoint_dir

os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(__file__, ckpt_dir)

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

def preprocess_dataset(csv_path, apply_zigzag=True, apply_direction=True):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    data_tensor_list = []
    label_tensor_list = []

    snake_index = get_snake_index_2d()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    for idx in range(len(df)):
        bytes_raw = ast.literal_eval(df.loc[idx, 'bytes'])            # list of 5×320
        dirs_raw = ast.literal_eval(df.loc[idx, 'direction_list'])    # list of 5

        packets = []
        for i in range(5):
            pkt = th.FloatTensor(bytes_raw[i])  # shape [320]

            if apply_zigzag:
                pkt = pkt[snake_index]

            pkt = pkt.view(8, 40)

            if apply_direction and dirs_raw[i] != 1:
                pkt = th.flip(pkt, dims=[-1])  # horizontal flip

            packets.append(pkt)

        img = th.cat(packets, dim=0).unsqueeze(0)  # [1, 40, 40]
        img = img / 255.0
        img = normalize(img)

        label = int(df.loc[idx, 'label'])
        data_tensor_list.append(img)
        label_tensor_list.append(label)

    data_tensor = th.stack(data_tensor_list)          # [N, 1, 40, 40]
    label_tensor = th.LongTensor(label_tensor_list)   # [N]
    return data_tensor, label_tensor

data_tensor, label_tensor = preprocess_dataset('../datasets/' + dataset + '.csv',
                                               apply_zigzag=False,
                                               apply_direction=False)

num_class = len(th.unique(label_tensor))

# 划分数据集
num_samples = data_tensor.shape[0]
train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size

train_data = th.utils.data.TensorDataset(data_tensor[:train_size], label_tensor[:train_size])
val_data   = th.utils.data.TensorDataset(data_tensor[train_size:train_size+val_size], label_tensor[train_size:train_size+val_size])
test_data  = th.utils.data.TensorDataset(data_tensor[train_size+val_size:], label_tensor[train_size+val_size:])

loader = {
    'train': th.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'val':   th.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False),
    'test':  th.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False),
}


model = TrafficTransformer(
        img_size=40, patch_size=2, in_chans=1, embed_dim=192, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_class)

model.load_state_dict(th.load(pretrained_model_path, map_location=gpu)['model'], strict=False)


# model.load_state_dict(th.load('./checkpoint/cstnet/checkpoint.pth', map_location=gpu), strict=False)

# ckpt = th.load('./checkpoint/cstnet/checkpoint.pth', map_location=gpu)
# model.pre_model.load_state_dict(ckpt['pre_model'], strict=True)


model = model.to(gpu)

# Training

optimizer = th.optim.Adam(model.parameters(), lr=mae_lr)

def train_step(engine, batch):
    global model, optimizer
    model.train()
    # model = model.to(gpu)
    optimizer.zero_grad()
    (data, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(data)
    y_true = label.type(th.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


trainer = Engine(train_step)

def test_step(engine, batch):
    global model
    with th.no_grad():
        model.eval()
        # model = model.to(gpu)
        (data, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(data)
        y_true = label
        return y_pred, y_true


evaluator = Engine(test_step)

recall = Recall(average='macro')
precision = Precision(average='macro')
F1 = (precision * recall * 2 / (precision + recall))
# F1 = Fbeta(beta=1.0, average='macro')

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
    evaluator.run(loader['train'])
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['val'])
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['test'])
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
                'pre_model': model.state_dict(),
                # 'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc
    # scheduler.step()

        
log_training_results.best_val_acc = 0
trainer.run(loader['train'], max_epochs=epochs)
