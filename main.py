import timm
import aros_node
import argparse
import torch
import torch.nn as nn
from aros_node.evaluate import *
from aros_node.utils import *
from tqdm import tqdm
from aros_node.data_loader import *
from aros_node.stability_loss_function import *
class ResNetWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        )

    def forward(self, x):

        x = (x - self.mean) / self.std

        return self.backbone(x)


def pretrain_model(model, train_loader, num_classes, epochs=5, lr=1e-4, device='cuda'):
    print(f"--- Starting Pre-tuning for {epochs} epochs ---")
    # Додаємо тимчасовий лінійний шар для класифікації (якщо backbone видає лише фічі)
    # feature_dim для resnet50 зазвичай 2048
    feature_dim = 2048
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            features = model(imgs)
            logits = classifier(features)
            loss = criterion(logits, lbls)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    return model

def main():

    print(f"Чи доступна CUDA: {torch.cuda.is_available()}")
    print(f"Поточний девайс: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    parser = argparse.ArgumentParser(description="Hyperparameters for the script")

    parser.add_argument('--fast', type=bool, default=True, help='Toggle between fast and full fake data generation modes')
    parser.add_argument('--epoch1', type=int, default=5, help='Number of epochs for stage 1')
    parser.add_argument('--epoch2', type=int, default=5, help='Number of epochs for stage 2')
    parser.add_argument('--epoch3', type=int, default=5, help='Number of epochs for stage 3')
    parser.add_argument('--in_dataset', type=str, default='eurosat', choices=['cifar10', 'cifar100', 'eurosat'], help='The in-distribution dataset to be used')
    parser.add_argument('--threat_model', type=str, default='Linf', help='Adversarial threat model for robust training')
    parser.add_argument('--noise_std', type=float, default=1, help='Standard deviation of noise for generating noisy fake embeddings')
    parser.add_argument('--attack_eps', type=float, default=8/255, help='Perturbation bound (epsilon) for PGD attack')
    parser.add_argument('--attack_steps', type=int, default=10, help='Number of steps for the PGD attack')
    parser.add_argument('--attack_alpha', type=float, default=2.5 * (8/255) / 10, help='Step size (alpha) for each PGD attack iteration')
    args = parser.parse_args('')

    # Set the default model name based on the selected dataset
    if args.in_dataset == 'eurosat':
        default_model_name = 'Cui2023Decoupled_WRN-28-10'
    elif args.in_dataset == 'cifar10':
        default_model_name = 'Rebuffi2021Fixing_70_16_cutmix_extra'
    elif args.in_dataset == 'cifar100':
        default_model_name = 'Wang2023Better_WRN-70-16'

    parser.add_argument('--model_name', type=str, default=default_model_name, choices=['Rebuffi2021Fixing_70_16_cutmix_extra', 'Wang2023Better_WRN-70-16','Cui2023Decoupled_WRN-28-10'], help='The pre-trained model to be used for feature extraction')

    # Re-parse arguments to include model_name selection based on the dataset
    args = parser.parse_args('')
    num_classes = 100 if args.in_dataset == 'cifar100' else 10

    trainloader, testloader,test_set, ID_OOD_loader = get_loaders(in_dataset=args.in_dataset)




    current_batch_size = trainloader.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    fake_loader=None

    backbone = timm.create_model(
        'resnet50.a1_in1k',
        pretrained=True,
        num_classes=0,
        global_pool='avg'
    )
    robust_backbone = ResNetWrapper(backbone).to(device)

    robust_backbone = pretrain_model(
        model=robust_backbone,
        train_loader=trainloader,
        num_classes=num_classes,
        epochs=5,
        device=device
    )

    num_fake_samples = len(trainloader.dataset) // num_classes

    x = torch.randn(2, 3, 64, 64).to(device)

    with torch.no_grad():
        y = robust_backbone(x)

    print(y.shape)


    embeddings, labels = [], []

    with torch.no_grad():
        for imgs, lbls in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            embed = robust_backbone(imgs).cpu()  # move to CPU only once per batch
            embeddings.append(embed)
            labels.append(lbls)
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()


    print("embedding computed...")


    if args.fast==False:
      gmm_dict = {}
      for cls in np.unique(labels):
          cls_embed = embeddings[labels == cls]
          gmm = GaussianMixture(n_components=1, covariance_type='full').fit(cls_embed)
          gmm_dict[cls] = gmm

      print("fake crafing...")

      fake_data = []


      for cls, gmm in gmm_dict.items():
          samples, likelihoods = [], []
          while len(samples) < num_fake_samples:
              s = gmm.sample(100)[0]
              likelihood = gmm.score_samples(s)
              samples.append(s[likelihood < np.quantile(likelihood, 0.001)])
              likelihoods.append(likelihood[likelihood < np.quantile(likelihood, 0.001)])
              if sum(len(smp) for smp in samples) >= num_fake_samples:
                  break
          samples = np.vstack(samples)[:num_fake_samples]
          fake_data.append(samples)

      fake_data = np.vstack(fake_data)
      fake_data = torch.tensor(fake_data).float()
      fake_data = F.normalize(fake_data, p=2, dim=1)
      fake_labels = torch.full((fake_data.shape[0],), 10)
      fake_loader = DataLoader(TensorDataset(fake_data, fake_labels), batch_size=current_batch_size, shuffle=True)

    if args.fast==True:
        noisy_embeddings = torch.tensor(embeddings) + args.noise_std * torch.randn_like(torch.tensor(embeddings))
        # Normalize Noisy Embeddings
        noisy_embeddings = F.normalize(noisy_embeddings, p=2, dim=1)[:len(trainloader.dataset)//num_classes]
        # Convert to DataLoader if needed
        fake_labels = torch.full((noisy_embeddings.shape[0],), num_classes)[:len(trainloader.dataset)//num_classes]
        fake_loader = DataLoader(TensorDataset(noisy_embeddings, fake_labels), batch_size=current_batch_size, shuffle=True)


    final_model = stability_loss_function_(trainloader, testloader, robust_backbone, num_classes, fake_loader, args)

 
    test_attack = PGD_AUC(final_model, eps=args.attack_eps, steps=args.attack_steps, alpha=args.attack_alpha, num_classes=num_classes)
    get_clean_AUC(final_model, ID_OOD_loader , device, num_classes)
    import gc

    # Перед початком оцінки (adv_auc = ...)
    torch.cuda.empty_cache()
    gc.collect()
    adv_auc = get_auc_adversarial(model=final_model,  test_loader=ID_OOD_loader, test_attack=test_attack, device=device, num_classes=num_classes)



if __name__ == "__main__":
    main()
'''
    model_dataset = 'cifar10' if args.in_dataset == 'eurosat' else args.in_dataset
    args.in_dataset = 'cifar10'
    # Завантажуємо модель, вказуючи підтримуваний датасет
    robust_backbone = aros_node.load_model(
        model_name=args.model_name,
        dataset=model_dataset,
        threat_model=args.threat_model
    ).to(device)


    last_layer_name, last_layer = list(robust_backbone.named_children())[-1]
    setattr(robust_backbone, last_layer_name, nn.Identity())'''
'''
    indices = torch.arange(500)  # Беремо лише 500 зображень
    train_set_small = Subset(trainloader.dataset, indices)
    trainloader = DataLoader(train_set_small, batch_size=64, shuffle=True)

    # Те саме для тестлоадера:
    indices_test = torch.arange(200)
    testloader = DataLoader(Subset(testloader.dataset, indices_test), batch_size=64)

'''