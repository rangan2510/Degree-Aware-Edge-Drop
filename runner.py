#%% ###############################################################################
#Imports
import dgl
import torch
import torch.nn.functional as F
import time
import numpy as np
import psutil
from models import Net
from log.logger import Logger
from log.summaryWriter import Summarize
dgl.seed(0)
torch.manual_seed(0)
dgl.random.seed(0)

#%% ###############################################################################
# Configuration
RUNS = 10
EDROPTYPE = 0
EDROPOUT = 0.5
EDROPLO = 0.01
EDROPHI = 0.9
NDROPOUT = 0.1
EPOCHS = 10
NORM = True
N_BLOCKS = 2

#Logger
model_name = "gcn" + str(EDROPTYPE) + "-L" + str(N_BLOCKS)

# Load data
from dataloader.CoraFull import load_data
g, features, labels, train_mask, test_mask, in_feat, out_feat = load_data()
IN_FEAT = in_feat
H_FEAT = 256
OUT_FEAT = out_feat


#%% ###############################################################################
# Eval
def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

#%% ###############################################################################
# Train
print("CPU cores:", psutil.cpu_count())

g = dgl.add_self_loop(g)

best_scores =[]
summary_path = None

for run in range(RUNS):
    RUN_ID = run+1
    print('-' ,RUN_ID, '-'*50)
    
    net = Net(g, IN_FEAT, H_FEAT, OUT_FEAT, N_BLOCKS, EDROPOUT, EDROPTYPE, EDROPLO, EDROPHI, NDROPOUT, NORM)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    dur = []
        
    best_score = 0
    best_epoch = 0

    logger = Logger(save_path="results",dataset_name="CoraFull", model_name=model_name, run_id=RUN_ID)
    logger.createDir()
    logger.start()
    summary_path = logger.path

    for epoch in range(EPOCHS):
        if epoch >=3:
            t0 = time.time()

        net.train()
        
        logits = net(g, features)

        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch >=3:
            dur.append(time.time() - t0)
        
        acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))

        logger.writelog("Epoch {:05d} , Loss {:.4f} , Test Acc {:.4f} , Time(s) {:.4f}".format(epoch, loss.item(), acc, np.mean(dur)))

        if best_score < acc:
            best_score = acc
            best_epoch = epoch

    print("Best Test Acc {:.4f} at Epoch {:05d}".format(best_score, best_epoch))
    best_scores.append(best_score)
    logger.saveModel(net)
    
    # savefile = str(DROPTYPE) + "_" + str(RUN_ID) + "_" + str(N_BLOCKS) + "_" + str(int(best_score*10**5)) + '.weights'
    # torch.save(net.state_dict(),savefile)
    
    del net
    del logger

summary = Summarize(best_scores, summary_path)
summary.writeSummary()