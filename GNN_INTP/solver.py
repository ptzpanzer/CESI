import importlib
import time
import json
import os

import torch
import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from Datasets import aio_dataloader as dl
import support_functions


def seed_worker(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)

    if not torch.cuda.is_available(): 
        torch.manual_seed(worker_id)
    else:
        torch.manual_seed(worker_id)
        torch.cuda.manual_seed(worker_id)
        torch.cuda.manual_seed_all(worker_id)


def training(settings):
    # merge settings
    solver_name = f"solver_{settings['model'].lower()}"
    try:
        my_model = importlib.import_module(f"solver_files.{solver_name}")
    except ImportError:
        raise ImportError(f"Model Module '{solver_name}' not found")
    settings = support_functions.merge_settings(my_model.required_settings, settings)
    print(f"Settings of training:")
    print(json.dumps(settings, indent=2, default=str))

    # set seed
    support_functions.seed_everything(settings['seed'])
    g = torch.Generator()
    g.manual_seed(settings['seed'])
    clip_models = ["Transformer_Stage", "Transformer_NA"]
    
    # Get device setting
    if not torch.cuda.is_available(): 
        device = torch.device("cpu")
        ngpu = 0
        print(f'Working on CPU')
        num_workers = 16
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            print(f'Working on multi-GPU {device_list}')
            num_workers = 16
        else:
            print(f'Working on single-GPU')
            num_workers = 10

    # build dataloader
    dataset_train = dl.IntpDataset(settings=settings, mask_distance=-1, call_name='train')
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=settings['real_batch'], shuffle=True, collate_fn=dataset_train.collate_fn, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, prefetch_factor=32, drop_last=True)

    test_dataloaders = []
    for mask_distance in [0, ]:
        this_dataset = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='test')
        test_dataloaders.append(
            torch.utils.data.DataLoader(
                this_dataset, 
                batch_size=settings['real_batch'], shuffle=False, 
                collate_fn=this_dataset.collate_fn, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, prefetch_factor=64, drop_last=True
            )
        )

    # build model
    model = my_model.INTP_Model(settings=settings, device=device)
    loss = model.loss_func
    optimizer = model.optimizer
    scheduler = model.scheduler
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    elif ngpu == 1:
        model = model.cuda()
    print(model)

    # set training loop
    epochs = settings['epoch']
    batch_size = settings['real_batch']
    settings['accumulation_steps'] = settings['full_batch'] // settings['real_batch']
    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, settings['full_batch']))

    # fire training loop
    start_time = time.time()
    list_total = []
    list_err = []
    best_loss = float('inf')
    es_counter = 0
    
    iter_counter = 0
    epoch_counter = 0
    epoch_loss = 0
    mini_loss = 0
    data_iter = iter(dataloader_tr)

    t_train_iter_start = time.time()
    while True:
        try:
            batch = next(data_iter)

            if ngpu == 1:
                batch = (item.to('cuda') for item in batch)

            # train 1 iteration
            model.train()
            real_iter = iter_counter // settings['accumulation_steps'] + 1
            output_tr, target_tr = model(batch)
            batch_loss, _, _ = loss(output_tr, target_tr)

            batch_loss /= settings['accumulation_steps']
            epoch_loss += batch_loss.item()
            mini_loss += batch_loss.item()
            batch_loss.backward()

            if (iter_counter+1) % settings['accumulation_steps'] == 0:
                if settings["model"] in clip_models:
                    clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # with open(settings['coffer_slot'] + "grads.txt", "a") as f:
                #     f.write(f'iter {real_iter}:\n')
                #     for n, p in model.named_parameters():
                #         if (p.requires_grad) and ("bias" not in n):
                #             f.write(f'\tave_grads {n}: {p.grad.abs().mean().detach().cpu().numpy()}\n')
                #             f.write(f'\tmax_grads {n}: {p.grad.abs().max().detach().cpu().numpy()}\n')
                #     f.write(f'\toutput_tr: {output_tr[0].detach().cpu().numpy()}\n')
                #     f.write(f'\ttarget_tr: {target_tr.detach().cpu().numpy()}\n')
                #     f.write('\n')
                
                optimizer.zero_grad()
                if scheduler != None:
                    scheduler.step()
                    this_lr = scheduler.get_last_lr()
                else:
                    this_lr = "No scheduler"
                print(f'\tIter {real_iter} - Loss: {mini_loss} - LR: {this_lr}', end="\r", flush=True)
                # print(f'\tIter {real_iter} - Loss: {mini_loss} - LR: {this_lr}', flush=True)
                mini_loss = 0
            iter_counter += 1

        except StopIteration:
            # epoch finished, test model
            model.eval()
            with torch.no_grad():
                output_list = []
                target_list = []
                test_loss = 0
                for dataloader_ex in test_dataloaders:
                    for test_batch in dataloader_ex:
                        if ngpu == 1:
                            test_batch = (item.cuda() for item in test_batch)
                        output_ex, target_ex = model(test_batch)
                        test_batch_loss, o, t = loss(output_ex, target_ex)
                        test_loss += test_batch_loss.item()
                        output_list.append(o.squeeze().detach().cpu())
                        target_list.append(t.squeeze().detach().cpu())
                output = torch.cat(output_list)
                target = torch.cat(target_list)

                tgt_min = test_dataloaders[0].dataset.dic_op_minmax[test_dataloaders[0].dataset.dataset_info["tgt_op"]][0]
                tgt_max = test_dataloaders[0].dataset.dic_op_minmax[test_dataloaders[0].dataset.dataset_info["tgt_op"]][1]
                test_means_origin = (output * (tgt_max - tgt_min)) + tgt_min
                test_y_origin = (target * (tgt_max - tgt_min)) + tgt_min

                err = mean_absolute_error(test_y_origin, test_means_origin)
                mape = mean_absolute_percentage_error(test_y_origin, test_means_origin)
                
                end_time = time.time()
                epoch_time = end_time - start_time
                
                print(f'\n\t\t--------\n\t\tEpoch: {str(epoch_counter)}, epoch_loss: {epoch_loss}\n\t\t--------\n')
                print(f'\t\t--------\n\t\ttest_loss: {str(test_loss)}, last best test_loss: {str(best_loss)}\n\t\t--------\n')
                print(f'\t\t--------\n\t\tMAPE: {str(mape)}, MAE: {str(err)}\n\t\t--------\n')
                print(f'\t\t--------\n\t\tEpoch_time: {str(epoch_time)}\n\t\t--------\n')
                
                if best_loss - test_loss > settings['es_mindelta']:
                    best_loss = test_loss
                    torch.save(model.state_dict(), settings['coffer_slot'] + "best_params")
                    support_functions.save_square_img(
                        contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
                        xlabel='targets_ex', ylabel='output_ex', 
                        savename=os.path.join(settings['coffer_slot'], f'test_{epoch_counter}'),
                        title=f"Fold{settings['fold']}_holdout{settings['holdout']}_Md_all: MAE {err:.2f} MAPE {mape:.2f}"
                    )
                    es_counter = 0
                else:
                    es_counter += 1
                    print(f"INFO: Early stopping counter {es_counter} of {settings['es_endure']}")
                    if es_counter >= settings['es_endure']:
                        print('INFO: Early stopping')
                        break

                list_err.append(float(test_loss))
                list_total.append(float(epoch_loss))
                epoch_loss = 0
                epoch_counter += 1
                start_time = end_time

            if epoch_counter > settings['epoch']:
                break

            # test finished, reset dataloader to top and grab a new batch
            data_iter = iter(dataloader_tr)

    
    return list_total, list_err


def evaluate(settings):
    # merge settings
    solver_name = f"solver_{settings['model'].lower()}"
    try:
        my_model = importlib.import_module(f"solver_files.{solver_name}")
    except ImportError:
        raise ImportError(f"Model Module '{solver_name}' not found")
    settings = support_functions.merge_settings(my_model.required_settings, settings)
    print(f"Settings of evaluating:")
    print(json.dumps(settings, indent=2, default=str))

    # set seed
    support_functions.seed_everything(settings['seed'])
    g = torch.Generator()
    g.manual_seed(settings['seed'])
    
    # Get device setting
    if not torch.cuda.is_available(): 
        device = torch.device("cpu")
        ngpu = 0
        print(f'Working on CPU')
        num_workers = 16
    else:
        device = torch.device("cuda")
        ngpu = torch.cuda.device_count()
        if ngpu > 1:
            device_list = [i for i in range(ngpu)]
            print(f'Working on multi-GPU {device_list}')
            num_workers = 16
        else:
            print(f'Working on single-GPU')
            num_workers = 10

    # build dataloader
    eval_dataloaders = []
    for mask_distance in [0, 20, 50]:
        this_dataset = dl.IntpDataset(settings=settings, mask_distance=mask_distance, call_name='eval')
        eval_dataloaders.append(
            torch.utils.data.DataLoader(
                this_dataset, 
                batch_size=settings['real_batch'], shuffle=False, 
                collate_fn=this_dataset.collate_fn, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, prefetch_factor=64, drop_last=True
            )
        )

    # build model
    model = my_model.INTP_Model(settings=settings, device=device)
    loss = model.loss_func
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_list)
    elif ngpu == 1:
        model = model.cuda()
    model.load_state_dict(torch.load(settings['coffer_slot'] + "best_params"))

    rtn_mae_list = []
    rtn_mape_list = []
    rtn_rsq_list = []
    output_list = []
    target_list = []
    for dataloader_ev in eval_dataloaders:
        # Eval batch
        model.eval()
        for eval_batch in dataloader_ev:
            if ngpu == 1:
                eval_batch = (item.cuda() for item in eval_batch)
            output_ex, target_ex = model(eval_batch)
            _, o, t = loss(output_ex, target_ex)
            output_list.append(o.squeeze().detach().cpu())
            target_list.append(t.squeeze().detach().cpu())
    output = torch.cat(output_list)
    target = torch.cat(target_list)

    tgt_min = eval_dataloaders[0].dataset.dic_op_minmax[eval_dataloaders[0].dataset.dataset_info["tgt_op"]][0]
    tgt_max = eval_dataloaders[0].dataset.dic_op_minmax[eval_dataloaders[0].dataset.dataset_info["tgt_op"]][1]
    test_means_origin = (output * (tgt_max - tgt_min)) + tgt_min
    test_y_origin = (target * (tgt_max - tgt_min)) + tgt_min

    mae = mean_absolute_error(test_y_origin, test_means_origin)
    mape = mean_absolute_percentage_error(test_y_origin, test_means_origin)
    rsq = r2_score(test_y_origin, test_means_origin)
    
    rtn_mae_list.append(float(mae))
    rtn_mape_list.append(float(mape))
    rtn_rsq_list.append(float(rsq))
    
    print(f'\t\t--------\n\t\tMAE: {str(mae)}, R2: {str(rsq)}, MAPE: {str(mape)}\n\t\t--------\n')
    print(f'\t\t--------\n\t\tDiffer: {test_means_origin.max() - test_means_origin.min()}, count: {test_y_origin.size(0)}\n\t\t--------\n')

    support_functions.save_square_img(
        contents=[test_y_origin.numpy(), test_means_origin.numpy()], 
        xlabel='targets_ex', ylabel='output_ex', 
        savename=os.path.join(settings['coffer_slot'], f'result'),
        title=f"Fold{settings['fold']}_holdout{settings['holdout']}: MAE {mae:.2f} R2 {rsq:.2f} MAPE {mape:.2f}"
    )
    targets_ex = test_y_origin.unsqueeze(1)
    output_ex = test_means_origin.unsqueeze(1)
    diff_ex = targets_ex - output_ex
    pd_out = pd.DataFrame(
        torch.cat(
            (targets_ex, output_ex, diff_ex), 1
        ).numpy()
    )
    pd_out.columns = ['Target', 'Output', 'Diff']
    pd_out.to_csv(os.path.join(settings['coffer_slot'], f'result.csv'), index=False)

    return rtn_mae_list, rtn_mape_list, rtn_rsq_list
    