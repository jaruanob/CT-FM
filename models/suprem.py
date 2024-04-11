import torch

def SuPreM_loader(model, ckpt_path, decoder=True):
    model_dict = torch.load(ckpt_path)['net']
    store_dict = model.state_dict()
    amount = 0
    if model.__class__.__name__ == "SegResNet":
        print('Loading SuPreM SegResNet backbone pretrained weights')
        decoder_keys = ["up_layers", "up_samples", "conv_final"]
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys() and 'conv_final.2.conv' not in new_key:
                if not decoder and any(decoder_key in new_key for decoder_key in decoder_keys):
                    continue
                store_dict[new_key] = model_dict[key]   
                amount += 1
    
    elif model.__class__.__name__ == "UNet3D":
        print('Loading SuPreM UNet backbone pretrained weights')
        decoder_keys = ["up_tr", "out_tr"]
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[2:])
            if new_key in store_dict.keys():
                if not decoder and any(decoder_key in new_key for decoder_key in decoder_keys):
                    continue
                store_dict[new_key] = model_dict[key]   
                amount += 1

    else:
        raise ValueError('Model not supported')
    
    model.load_state_dict(store_dict)
    print(f"Loaded {amount}/{len(store_dict.keys())} keys")
    return model