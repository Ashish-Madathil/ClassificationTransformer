import torch
import torch.nn as nn
import argparse,math,numpy as np
from load_data import get_data
from models import CTranModel
from models import CTranModelCub
from config_args import get_args
import utils.evaluate as evaluate
import utils.logger as logger
from pdb import set_trace as stop
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import pandas as pd

args = get_args(argparse.ArgumentParser())


print('Labels: {}'.format(args.num_labels))
print('Train Known: {}'.format(args.train_known_labels))
print('Test Known:  {}'.format(args.test_known_labels))

train_loader,valid_loader,test_loader = get_data(args)


def save_predictions_to_csv(predictions, csv_filename='predictions_exp.csv'):
    # predictions_df = pd.DataFrame(predictions, columns=['Filename', 'EXP_Pred', 'ICM_Pred', 'TE_Pred', 'GT'])
    predictions_df = pd.DataFrame(predictions, columns=['Filename', 'EXP_Pred', 'GT'])
    predictions_df.to_csv(csv_filename, index=False)


if args.dataset == 'cub':
    model = CTranModelCub(args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)
    print(model.self_attn_layers)
else:
    model = CTranModel(args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)
    print(model.self_attn_layers)


def load_saved_model(saved_model_name,model):
    checkpoint = torch.load(saved_model_name)
    print(checkpoint.keys()) 
    model.load_state_dict(checkpoint['state_dict'])
    return model

print(args.model_name)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.cuda()

if args.inference:
    # model = load_saved_model(args.saved_model_name,model)
    model.load_state_dict(torch.load('best_model_checkpoint_exp.pth'))
    if test_loader is not None:
        data_loader =test_loader
    else:
        data_loader =valid_loader
    
    all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,data_loader,None,1,'Testing')
    all_preds_csv = all_preds.clone()
    all_preds_csv = torch.sigmoid(all_preds_csv)
    all_preds_csv = all_preds_csv.squeeze(0)
    print(all_preds.shape, all_preds.size(2))
    # all_preds_csv[:,0] = all_preds_csv[:,0] * 4
    # all_preds_csv[:,1:] = all_preds_csv[:,1:] * 3
    all_preds_csv = torch.round(all_preds_csv).int()

    # all_preds_csv[:,0] = torch.argmax(all_preds_csv[:,0])
    # all_preds_csv[:,1:] = all_preds_csv[:,1:].argmax().item()
    # all_preds_csv[:,1:] = torch.argmax(all_preds_csv[:,1:])

    # Assuming all_ids contains the image filenames
    extracted_predictions = []  # Extract and structure the predictions here
    for idx, filename in enumerate(all_ids):
        # extracted_predictions.append((filename, all_preds[idx], all_targs[idx]))
        # extracted_predictions.append((filename, all_preds_csv[idx, 0].item(), all_preds_csv[idx, 1].item(), all_preds_csv[idx, 2].item(), all_targs[idx].int().tolist()))
        extracted_predictions.append((filename, all_preds_csv[idx], all_targs[idx].int().tolist()))

    # Save the predictions to a CSV file
    save_predictions_to_csv(extracted_predictions)    
    test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)

    exit(0)

if args.freeze_backbone:
    for p in model.module.backbone.parameters():
        p.requires_grad=False
    for p in model.module.backbone.base_network.layer4.parameters():
        p.requires_grad=True

if args.optim == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)#, weight_decay=0.0004) 
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)

if args.warmup_scheduler:
    step_scheduler = None
    scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
else:
    scheduler_warmup = None
    if args.scheduler_type == 'plateau':
        step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)
    elif args.scheduler_type == 'step':
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        step_scheduler = None

metrics_logger = logger.Logger(args)
loss_logger = logger.LossLogger(args.model_name)

#NEWLY ADDED
best_valid_loss = float('inf')
best_valid_epoch = 0
patience = 50  # Number of epochs to wait for improvement before stopping
###

for epoch in range(1,args.epochs+1):
    print('======================== {} ========================'.format(epoch))
    for param_group in optimizer.param_groups:
        print('LR: {}'.format(param_group['lr']))

    train_loader.dataset.epoch = epoch
    ################### Train #################
    all_preds,all_targs,all_masks,all_ids,train_loss,train_loss_unk = run_epoch(args,model,train_loader,optimizer,epoch,'Training',train=True,warmup_scheduler=scheduler_warmup)
    # print('all_preds : ',all_preds.shape, all_preds)
    # print('all_targs : ',all_targs.shape, all_targs, all_targs.shape[1])
    # print('all_masks : ',all_masks.shape, all_masks)
    train_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,train_loss,train_loss_unk,0,args.train_known_labels)
    loss_logger.log_losses('train.log',epoch,train_loss,train_metrics,train_loss_unk)

    ################### Valid #################
    all_preds,all_targs,all_masks,all_ids,valid_loss,valid_loss_unk = run_epoch(args,model,valid_loader,None,epoch,'Validating')
    valid_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,valid_loss,valid_loss_unk,0,args.test_known_labels)
    loss_logger.log_losses('valid.log',epoch,valid_loss,valid_metrics,valid_loss_unk)

# NEWLY ADDED
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_valid_epoch = epoch
        # Save the model checkpoint whenever validation loss improves
        torch.save(model.state_dict(), 'best_model_checkpoint_exp.pth')
        # torch.save({'state_dict': model.state_dict()}, 'best_model_checkpoint.pth')

    # Check if training should be stopped based on patience
    if epoch - best_valid_epoch > patience:
        print(f"Validation loss hasn't improved for {patience} epochs. Stopping training.")
        break
###

    ################### Test #################
    if test_loader is not None:
        all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,test_loader,None,epoch,'Testing')
        test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)
    else:
        test_loss,test_loss_unk,test_metrics = valid_loss,valid_loss_unk,valid_metrics
    loss_logger.log_losses('test.log',epoch,test_loss,test_metrics,test_loss_unk)

    if step_scheduler is not None:
        if args.scheduler_type == 'step':
            step_scheduler.step(epoch)
        elif args.scheduler_type == 'plateau':
            step_scheduler.step(valid_loss_unk)

    ############## Log and Save ##############
    best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_preds,all_targs,all_ids,args)

    print(args.model_name)
