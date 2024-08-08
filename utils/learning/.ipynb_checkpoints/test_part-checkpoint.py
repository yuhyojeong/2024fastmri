import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
#from utils.data.load_data2 import create_data_loaders
from utils.data.load_data import create_data_loaders
#from utils.model.nafvarnet_copy import VarNet
#from utils.model.varnet_nafssr import VarNet
# from utils.model.promptmr import VarNet
from utils.model.varnet import VarNet


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, grappa, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)
            
            output = model(kspace, mask, grappa)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)

#     model = VarNet(num_cascades= 10,
#         num_adj_slices= 1, #5
#         n_feat0 = 20, #48
#         feature_dim=[4, 8, 12],
#         prompt_dim=[2, 4, 8],
#         sens_n_feat0=10,
#         sens_feature_dim= [4, 8, 12],
#         sens_prompt_dim= [4, 8, 12],
#         len_prompt= [4, 4, 4],
#         prompt_size=[16, 8, 4],
#         n_enc_cab= [2, 2, 3],
#         n_dec_cab= [2, 2, 3],
#         n_skip_cab= [1, 1, 1],
#         n_bottleneck_cab= 1,
#         no_use_ca= False,
#         sens_len_prompt= None,
#         sens_prompt_size= None,
#         sens_n_enc_cab= None,
#         sens_n_dec_cab = None,
#         sens_n_skip_cab = None,
#         sens_n_bottleneck_cab= None,
#         sens_no_use_ca= None,
#         mask_center=True,
#         use_checkpoint=True,
#         low_mem=False,)
    
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)