import torch

def SuPreM_loader(model, ckpt_path):
    model_dict = torch.load(ckpt_path)['net']
    store_dict = model.state_dict()
    amount = 0
    if model.__class__.__name__ == "SegResNet":
        print('Loading SuPreM SegResNet backbone pretrained weights')

        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys() and 'conv_final.2.conv' not in new_key:
                store_dict[new_key] = model_dict[key]   
                amount += 1
    
    elif model.__class__.__name__ == "UNet3D":
        print('Loading SuPreM UNet backbone pretrained weights')
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[2:])
            if new_key in store_dict.keys():
                store_dict[new_key] = model_dict[key]   
                amount += 1

    else:
        raise ValueError('Model not supported')
    
    model.load_state_dict(store_dict)
    print(amount, len(store_dict.keys()))
    return model