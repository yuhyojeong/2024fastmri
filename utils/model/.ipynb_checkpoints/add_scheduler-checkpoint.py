import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from promptmr_nafnet import VarNet


# Load your existing checkpoint
checkpoint_path = '../../../result/test_Varnet/checkpoints/model.pt'
checkpoint = torch.load(checkpoint_path)

# Re-initialize the model (adjust the initialization as per your model's configuration)
model = VarNet(num_cascades= 10,
        num_adj_slices= 1, #5
        n_feat0 = 20, #48
        feature_dim=[4, 8, 12],
        prompt_dim=[2, 4, 8],
        sens_n_feat0=10,
        sens_feature_dim= [4, 8, 12],
        sens_prompt_dim= [4, 8, 12],
        len_prompt= [4, 4, 4],
        prompt_size=[16, 8, 4],
        n_enc_cab= [2, 2, 3],
        n_dec_cab= [2, 2, 3],
        n_skip_cab= [1, 1, 1],
        n_bottleneck_cab= 1,
        no_use_ca= False,
        sens_len_prompt= None,
        sens_prompt_size= None,
        sens_n_enc_cab= None,
        sens_n_dec_cab = None,
        sens_n_skip_cab = None,
        sens_n_bottleneck_cab= None,
        sens_no_use_ca= None,
        mask_center=True,
        use_checkpoint=True,
        low_mem=False,)

# Re-initialize the optimizer with the same parameters
optimizer = torch.optim.AdamW(model.parameters(), 0.0007)

# Initialize the scheduler with the same settings
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

# If you know the last validation loss, set it to the scheduler
scheduler.step(checkpoint['best_val_loss'])

# Add the scheduler state to the checkpoint
checkpoint['scheduler'] = scheduler.state_dict()

# Save the updated checkpoint
torch.save(checkpoint, checkpoint_path)

print("Scheduler state added to the checkpoint successfully.")
