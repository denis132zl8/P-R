import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve
)

from sklearn.decomposition import PCA

from torch.autograd.functional import jvp

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", DEVICE)

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 128
LATENT = 128

EPOCHS = 10
LR = 1e-3

MAX_EVAL = 2000

# losses
LAMBDA_EQ = 0.1
LAMBDA_SEP = 0.5
LAMBDA_LYAP = 0.01
LAMBDA_ADV = 0.2

# PGD
PGD_STEPS = 3
PGD_ALPHA = 0.01
PGD_EPS = 0.03

# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(
    "./data",
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    "./data",
    train=False,
    download=True,
    transform=transform
)

ood_svhn = torchvision.datasets.SVHN(
    "./data",
    split='test',
    download=True,
    transform=transform
)

ood_cifar100 = torchvision.datasets.CIFAR100(
    "./data",
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(trainset, range(20000)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(testset, range(MAX_EVAL)),
    batch_size=BATCH_SIZE
)

svhn_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(ood_svhn, range(MAX_EVAL)),
    batch_size=BATCH_SIZE
)

cifar100_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(ood_cifar100, range(MAX_EVAL)),
    batch_size=BATCH_SIZE
)

# ============================================================
# NODE BLOCK
# ============================================================

class NODEBlock(nn.Module):

    def __init__(self, dim, steps=5, dt=0.2):

        super().__init__()

        self.steps = steps
        self.dt = dt

        self.f = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):

        z = x

        traj = [z]

        for _ in range(self.steps):

            z = z + self.dt * self.f(z)

            traj.append(z)

        return z, traj

# ============================================================
# MODEL
# ============================================================

class HybridOOD(nn.Module):

    def __init__(self, latent=128, classes=10):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4,4))
        )

        self.fc = nn.Linear(128*4*4, latent)

        self.node = NODEBlock(latent)

        self.classifier = nn.Linear(latent, classes)

        self.separator = nn.Sequential(
            nn.Linear(latent, latent),
            nn.ReLU(),
            nn.Linear(latent, 1)
        )

    def forward(self, x, return_traj=False):

        states = []

        x = self.encoder[0](x)
        x = self.encoder[1](x)
        states.append(x)

        x = self.encoder[2](x)

        x = self.encoder[3](x)
        x = self.encoder[4](x)
        states.append(x)

        x = self.encoder[5](x)

        x = self.encoder[6](x)
        x = self.encoder[7](x)
        states.append(x)

        x = self.encoder[8](x)

        x = x.flatten(1)

        z = F.relu(self.fc(x))

        z_eq, traj = self.node(z)

        logits = self.classifier(z_eq)

        sep = self.separator(z_eq)

        if return_traj:
            return logits, sep, z_eq, traj, states

        return logits, sep, z_eq

# ============================================================
# LOSSES
# ============================================================

def equilibrium_loss(traj):

    loss = 0.0

    for i in range(len(traj)-1):

        loss += (
            traj[i+1] - traj[i]
        ).pow(2).mean()

    return loss

# ------------------------------------------------------------

def separator_loss(sep_id, sep_fake):

    labels_id = torch.ones_like(sep_id)

    labels_fake = torch.zeros_like(sep_fake)

    return (
        F.binary_cross_entropy_with_logits(
            sep_id,
            labels_id
        )
        +
        F.binary_cross_entropy_with_logits(
            sep_fake,
            labels_fake
        )
    )

# ------------------------------------------------------------

def lyapunov_regularization(states):

    reg = 0.0

    for s in states:

        flat = s.reshape(s.size(0), -1)

        variance = torch.var(flat, dim=1).mean()

        reg += variance

    return reg

# ============================================================
# SYNTHETIC LATENT OOD
# ============================================================

def generate_fake_ood(z, noise_scale=3.0):

    noise = torch.randn_like(z)

    fake = z + noise_scale * noise

    return fake

# ============================================================
# PGD ATTACK
# ============================================================

def pgd_attack(model, x, y):

    x_adv = x.clone().detach()

    x_adv += torch.empty_like(x_adv).uniform_(
        -PGD_EPS,
        PGD_EPS
    )

    for _ in range(PGD_STEPS):

        x_adv.requires_grad_(True)

        logits, _, _ = model(x_adv)

        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(
            loss,
            x_adv
        )[0]

        x_adv = x_adv.detach() + PGD_ALPHA * grad.sign()

        delta = torch.clamp(
            x_adv - x,
            -PGD_EPS,
            PGD_EPS
        )

        x_adv = (x + delta).detach()

    return x_adv

# ============================================================
# SCORES
# ============================================================

def msp_score(logits):

    probs = F.softmax(logits, dim=1)

    return torch.max(probs, dim=1).values

# ------------------------------------------------------------

def energy_score(logits):

    # FIXED SIGN

    return torch.logsumexp(logits, dim=1)

# ------------------------------------------------------------

def compute_lyapunov(model, x, samples=5):

    x = x.clone().detach().requires_grad_(True)

    values = []

    for _ in range(samples):

        def f(inp):

            _, _, z = model(inp)

            return z

        v = torch.randn_like(x)

        v = v / (torch.norm(v) + 1e-8)

        _, jv = jvp(
            f,
            (x,),
            (v,),
            create_graph=False
        )

        val = torch.log(
            torch.norm(jv) + 1e-8
        )

        values.append(val.item())

    return np.mean(values)

# ============================================================
# METRICS
# ============================================================

def compute_fpr95(id_scores, ood_scores):

    threshold = np.percentile(id_scores, 5)

    return np.mean(ood_scores >= threshold)

# ------------------------------------------------------------

def compute_metrics(id_scores, ood_scores):

    labels = np.concatenate([
        np.ones_like(id_scores),
        np.zeros_like(ood_scores)
    ])

    scores = np.concatenate([
        id_scores,
        ood_scores
    ])

    auroc = roc_auc_score(labels, scores)

    aupr = average_precision_score(labels, scores)

    fpr95 = compute_fpr95(
        id_scores,
        ood_scores
    )

    fpr, tpr, _ = roc_curve(
        labels,
        scores
    )

    return auroc, aupr, fpr95, fpr, tpr

# ============================================================
# EVALUATION
# ============================================================

def collect_scores(loader, model):

    model.eval()

    sep_scores = []
    msp_scores_all = []
    energy_scores_all = []
    lyap_scores = []

    embeddings = []

    count = 0

    with torch.no_grad():

        for x, _ in loader:

            x = x.to(DEVICE)

            logits, sep, z = model(x)

            sep_scores.extend(
                torch.sigmoid(sep)
                .cpu()
                .numpy()
                .flatten()
            )

            msp_scores_all.extend(
                msp_score(logits)
                .cpu()
                .numpy()
            )

            energy_scores_all.extend(
                energy_score(logits)
                .cpu()
                .numpy()
            )

            embeddings.append(
                z.cpu().numpy()
            )

            count += x.size(0)

            if count >= MAX_EVAL:
                break

    # Separate Lyapunov estimation

    for x, _ in loader:

        x = x.to(DEVICE)

        for i in range(x.size(0)):

            val = compute_lyapunov(
                model,
                x[i:i+1]
            )

            lyap_scores.append(val)

            if len(lyap_scores) >= MAX_EVAL:
                break

        if len(lyap_scores) >= MAX_EVAL:
            break

    return (
        np.array(sep_scores),
        np.array(msp_scores_all),
        np.array(energy_scores_all),
        np.array(lyap_scores),
        np.concatenate(embeddings)
    )

# ============================================================
# TRAIN
# ============================================================

model = HybridOOD().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

loss_curve = []

print("\nTraining...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for x, y in trainloader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # adversarial examples
        x_adv = pgd_attack(
            model,
            x,
            y
        )

        optimizer.zero_grad()

        logits, sep, z, traj, states = model(
            x,
            return_traj=True
        )

        logits_adv, _, _, _, _ = model(
            x_adv,
            return_traj=True
        )

        # synthetic fake OOD
        z_fake = generate_fake_ood(z)

        sep_fake = model.separator(z_fake)

        # losses
        cls_loss = F.cross_entropy(
            logits,
            y
        )

        adv_loss = F.cross_entropy(
            logits_adv,
            y
        )

        eq_loss = equilibrium_loss(traj)

        sep_loss = separator_loss(
            sep,
            sep_fake
        )

        lyap_loss = lyapunov_regularization(
            states
        )

        loss = (
            cls_loss
            +
            LAMBDA_ADV * adv_loss
            +
            LAMBDA_EQ * eq_loss
            +
            LAMBDA_SEP * sep_loss
            +
            LAMBDA_LYAP * lyap_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    loss_curve.append(total_loss)

    print(
        f"Epoch {epoch+1}: "
        f"Loss = {total_loss:.3f}"
    )

# ============================================================
# EVALUATION
# ============================================================

print("\nCollecting scores...\n")

id_scores = collect_scores(
    testloader,
    model
)

svhn_scores = collect_scores(
    svhn_loader,
    model
)

cifar100_scores = collect_scores(
    cifar100_loader,
    model
)

# ============================================================
# RESULTS
# ============================================================

METHODS = [
    "SEP",
    "MSP",
    "ENERGY",
    "LYAP"
]

for dataset_name, ood_scores in [

    ("SVHN", svhn_scores),

    ("CIFAR100", cifar100_scores)

]:

    print("\n================================================")
    print(dataset_name)
    print("================================================")

    for i, method in enumerate(METHODS):

        auroc, aupr, fpr95, fpr, tpr = compute_metrics(
            id_scores[i],
            ood_scores[i]
        )

        print(
            f"{method}: "
            f"AUROC={auroc:.4f} "
            f"AUPR={aupr:.4f} "
            f"FPR95={fpr95:.4f}"
        )

        # ROC

        plt.figure()

        plt.plot(fpr, tpr)

        plt.xlabel("FPR")
        plt.ylabel("TPR")

        plt.title(
            f"{dataset_name} - {method}"
        )

        plt.show()

        # HIST

        plt.figure()

        plt.hist(
            id_scores[i],
            bins=40,
            alpha=0.5,
            label="ID"
        )

        plt.hist(
            ood_scores[i],
            bins=40,
            alpha=0.5,
            label="OOD"
        )

        plt.legend()

        plt.title(
            f"{dataset_name} - {method}"
        )

        plt.show()

# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nPCA visualization...\n")

pca = PCA(n_components=2)

X = np.concatenate([
    id_scores[4],
    svhn_scores[4]
])

y = np.concatenate([
    np.ones(len(id_scores[4])),
    np.zeros(len(svhn_scores[4]))
])

X2 = pca.fit_transform(X)

plt.figure()

plt.scatter(
    X2[:,0],
    X2[:,1],
    c=y,
    s=5
)

plt.title(
    "Latent PCA (ID=1, OOD=0)"
)

plt.show()

# ============================================================
# LOSS CURVE
# ============================================================

plt.figure()

plt.plot(loss_curve)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()

print("\nDONE.")