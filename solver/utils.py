import torch
import torch.nn as nn
import torch.nn.functional as F

# mask size: (c_in*k*k, c_o), fisher size: (c_in*k*k, c_o)
def get_nm_mask(fisher, n, m):
    length = fisher.numel()
    group = int(length / m)

    fisher_tmp = fisher.detach().reshape(group, m)
    index = torch.argsort(fisher_tmp, dim=1)[:, :int(m-n)]

    mask = torch.ones_like(fisher_tmp)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(fisher.shape)
    return mask # mask: (c_in*k*k, co)

@torch.no_grad()
def get_fisher(model, train_data, num_samples, param_groups):
    fisher_dict = {}

    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=1,
        num_workers=4,
        shuffle=True,
    )
    with torch.enable_grad():
        for idx, (image, label) in enumerate(train_dataloader):
            if idx >= num_samples: break
            image, label = image.cuda(), label.cuda()

            output = model(image)
            loss = criterion(output, label)
            loss.backward()

            for group in param_groups:
                for p in group["params"]:
                    fisher_dict[id(p)] = fisher_dict.get(id(p), torch.zeros_like(p, requires_grad=False)) + torch.square(p.grad).data
            model.zero_grad()
    return fisher_dict