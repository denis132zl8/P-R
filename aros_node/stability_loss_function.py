import torch
from robustbench.utils import load_model
import torch.nn as nn
from torch.nn.parameter import Parameter
import aros_node.utils
from aros_node.utils import *
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, SubsetRandomSampler, ConcatDataset
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# Налаштування шляхів
robust_feature_savefolder = './CIFAR100_resnet_Nov_1'
train_savepath = './CIFAR100_train_resnetNov1.npz'
test_savepath = './CIFAR100_test_resnetNov1.npz'
ODE_FC_save_folder = './CIFAR100_resnet_Nov_1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ODE_FC_odebatch = 100


def stability_loss_function_(trainloader, testloader, robust_backbone, class_numbers, fake_loader, args):
    # Ініціалізація шарів
    robust_backbone_fc_features = MLP_OUT_ORTH1024(2048)
    fc_layers_phase1 = MLP_OUT_BALL(class_numbers)

    for param in fc_layers_phase1.parameters():
        param.requires_grad = False

    net_save_robustfeature = nn.Sequential(robust_backbone, robust_backbone_fc_features, fc_layers_phase1).to(device)

    for param in robust_backbone.parameters():
        param.requires_grad = False

    print(net_save_robustfeature)
    net_save_robustfeature = net_save_robustfeature.to(device)

    optimizer1 = torch.optim.Adam(net_save_robustfeature.parameters(), lr=5e-3, eps=1e-2, amsgrad=True)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=0.5)

    def train_save_robustfeature(epoch):
        criterion = nn.CrossEntropyLoss()
        print('\nEpoch: %d' % epoch)
        net_save_robustfeature.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer1.zero_grad()
            outputs = net_save_robustfeature(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer1.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        scheduler1.step()

    def test_save_robustfeature(epoch):
        best_acc = 0
        criterion = nn.CrossEntropyLoss()
        net_save_robustfeature.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net_save_robustfeature(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                             (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {'net_save_robustfeature': net_save_robustfeature.state_dict(), 'acc': acc, 'epoch': epoch}
            torch.save(state, robust_feature_savefolder + '/ckpt.pth')
            best_acc = acc

        save_training_feature(net_save_robustfeature, trainloader, fake_embeddings_loader=fake_loader)
        print('----')
        save_testing_feature(net_save_robustfeature, testloader)
        print('------------')

    # Фаза 1: Тренування ознак
    makedirs(robust_feature_savefolder)
    for epoch in range(0, args.epoch1):
        train_save_robustfeature(epoch)
        test_save_robustfeature(epoch)
    print('save robust feature to ' + robust_feature_savefolder)

    # Регуляризатори для ODE
    def df_dz_regularizer(odefunc, z):
        regu_diag = 0.
        regu_offdiag = 0.0
        # Змінні numm, time_df, exponent, trans тощо мають бути визначені в args або глобально
        for ii in np.random.choice(z.shape[0], min(numm, z.shape[0]), replace=False):
            batchijacobian = torch.autograd.functional.jacobian(
                lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii + 1, ...], create_graph=True
            )
            batchijacobian = batchijacobian.view(z.shape[1], -1)
            if batchijacobian.shape[0] != batchijacobian.shape[1]:
                raise Exception("wrong dim in jacobian")

            tempdiag = torch.diagonal(batchijacobian, 0)
            regu_diag += torch.exp(exponent * (tempdiag + trans))
            offdiat = torch.sum(
                torch.abs(batchijacobian) * ((-1 * torch.eye(batchijacobian.shape[0]).to(device) + 0.5) * 2), dim=0)
            off_diagtemp = torch.exp(exponent_off * (offdiat + transoffdig))
            regu_offdiag += off_diagtemp
        return regu_diag / numm, regu_offdiag / numm

    def f_regularizer(odefunc, z):
        tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
        regu_f = torch.pow(exponent_f * tempf, 2)
        return regu_f

    # Фаза 2: Тренування ODE блоку
    makedirs(ODE_FC_save_folder)
    odefunc = ODEfunc_mlp(0)
    feature_layers_ode = ODEBlocktemp(odefunc)
    fc_layers_ode = MLP_OUT_LINEAR(class_numbers)
    for param in fc_layers_ode.parameters():
        param.requires_grad = False

    ODE_FCmodel = nn.Sequential(feature_layers_ode, fc_layers_ode).to(device)

    train_loader_ODE = DataLoader(DenseDatasetTrain(), batch_size=ODE_FC_odebatch, shuffle=True, num_workers=2)
    batches_per_epoch = len(train_loader_ODE)
    data_gen = inf_generator(train_loader_ODE)

    optimizer2 = torch.optim.Adam(ODE_FCmodel.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.5)

    for epoch in range(args.epoch2):
        with tqdm(total=batches_per_epoch, desc=f"Epoch {epoch + 1} ODE training") as pbar:
            for itr in range(batches_per_epoch):
                optimizer2.zero_grad()
                x, y = data_gen.__next__()
                x = x.to(device)

                y0 = x
                # Прохід через ODEBlock (перший елемент Sequential)
                y1 = ODE_FCmodel[0](y0)

                regu1, regu2 = df_dz_regularizer(odefunc, y0)
                regu1 = regu1.mean()
                regu2 = regu2.mean()
                regu3 = f_regularizer(odefunc, y0).mean()

                loss = weight_f * regu3 + weight_diag * regu1 + weight_offdiag * regu2
                loss.backward()
                optimizer2.step()

                torch.cuda.empty_cache()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        scheduler2.step()
        print(f"Epoch {epoch + 1}, Learning Rate: {optimizer2.param_groups[0]['lr']}")

    # Фаза 3: Фінальне тренування всієї моделі
    feature_layers_final = ODEBlock(odefunc)
    fc_layers_final = MLP_OUT_LINEAR(class_numbers)
    ODE_FCmodel_final = nn.Sequential(feature_layers_final, fc_layers_final).to(device)

    for param in odefunc.parameters():
        param.requires_grad = True
    for param in robust_backbone_fc_features.parameters():
        param.requires_grad = False
    for param in robust_backbone.parameters():
        param.requires_grad = False

    new_model_full = nn.Sequential(robust_backbone, robust_backbone_fc_features, ODE_FCmodel_final).to(device)

    optimizer3 = torch.optim.Adam([
        {'params': odefunc.parameters(), 'lr': 1e-5, 'eps': 1e-6},
        {'params': fc_layers_final.parameters(), 'lr': 5e-3, 'eps': 1e-4}
    ], amsgrad=True)

    def train(net, epoch):
        criterion = nn.CrossEntropyLoss()
        print('\nFinal Training Epoch: %d' % epoch)
        net.train()
        fake_iter = iter(fake_loader)
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            try:
                fake_inputs, _ = next(fake_iter)
            except StopIteration:
                fake_iter = iter(fake_loader)
                fake_inputs, _ = next(fake_iter)

            fake_inputs = fake_inputs.to(device)
            optimizer3.zero_grad()

            # ID дані
            outputs_id = net(inputs)
            loss_id = criterion(outputs_id, targets)

            # OOD дані (минаючи backbone)
            out_fake = net[1](fake_inputs)
            out_fake = net[2](out_fake)

            # KL Divergence до рівномірного розподілу для OOD
            loss_ood = -torch.mean(torch.sum(F.log_softmax(out_fake, dim=1) / class_numbers, dim=1))

            loss = loss_id + 0.5 * loss_ood
            loss.backward()
            optimizer3.step()

            train_loss += loss.item()
            _, predicted = outputs_id.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%%' % (train_loss / (batch_idx + 1), 100. * correct / total))

    for epoch in range(0, args.epoch3):
        train(new_model_full, epoch)

    return new_model_full