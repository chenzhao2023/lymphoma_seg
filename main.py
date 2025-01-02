import os
import yaml
import time
import torch
import numpy as np
from tqdm import tqdm
from loss import DiceBCELoss
from dataloader import get_fold_dataloader
from monai.inferers import sliding_window_inference
from utils import compute_metrics, create_experiment_dir
from Configs import NetConfig as NET_CONFIG
from network import OurMethod


def run_fold_training(config):
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    # data loader
    fold_dir = os.path.join(config['data_root'], str(f"fold_{config['fold_id']}"))
    train_loader, valid_loader, test_loader = get_fold_dataloader(fold_dir, config['data_root'],
                                                                  config['batch_size'], config['num_workers'])

    device = torch.device(f"cuda:{config['cuda_id']}" if torch.cuda.is_available() else "cpu")
    net_work_config = NET_CONFIG()
    model = OurMethod(net_work_config).to(device)
    # loss and optimizer
    loss_function = DiceBCELoss(dice_weight=config['dice_weight'], bce_weight=config['bce_weight'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_decay_epoch'])

    best_dice = -1
    best_epoch = -1

    best_model_path = ''
    epochs_no_improve = 0  # early stopping counter

    # create experiment directory
    dirs = create_experiment_dir(config['save_root'], config['fold_id'], current_time)
    logger = dirs["logger"]

    logger.info("Training started")
    logger.info(f"Current time: {current_time}")
    logger.info(f"Configuration: {config}")
    try:

        for epoch in range(config['max_epochs']):
            model.train()

            logger.info(f"Epoch {epoch + 1}/{config['max_epochs']}")
            start_time = time.time()
            train_loss = 0
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch_data in tepoch:
                    # [batch, channel=1, depth=32, height=224, width=224]
                    ct = batch_data["CT"]
                    pt = batch_data["PET"]
                    labels = batch_data["MASK"]
                    # [batch, channel=2, depth=32, height=224, width=224]
                    inputs = torch.cat([ct, pt], dim=1).to(device)

                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        logger.error("Outputs contain NaN or Inf!")
                        raise ValueError("Model outputs contain invalid values.")

                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    del inputs, labels, outputs
                    torch.cuda.empty_cache()

            train_loss /= len(train_loader)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} Average Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")
            scheduler.step()

            if (epoch + 1) % config['val_interval'] == 0:
                model.eval()
                val_labels_list = []
                val_outputs_list = []

                with torch.no_grad():
                    for val_data in valid_loader:
                        ct = val_data["CT"]
                        pt = val_data["PET"]
                        val_labels = val_data["MASK"]
                        val_inputs = torch.cat([ct, pt], dim=1).to(device)

                        val_labels = val_labels.to(device)
                        val_outputs = sliding_window_inference(val_inputs, (32, 224, 224), 4, model)
                        val_outputs = torch.sigmoid(val_outputs)
                        val_outputs_list.append(val_outputs)
                        val_labels_list.append(val_labels)

                        del val_inputs, val_labels, val_outputs
                        torch.cuda.empty_cache()

                    val_outputs_list = torch.cat(val_outputs_list)
                    val_labels_list = torch.cat(val_labels_list)
                    # computer and logging metrics
                    current_dice, precision, recall = compute_metrics(val_outputs_list, val_labels_list)
                    logger.info(f"Validation Dice Score: {current_dice:.4f}, Precision: {precision:.4f}, "
                                f"Recall: {recall:.4f}")

                    if current_dice > best_dice:
                        # logging best performance
                        best_dice = current_dice
                        best_epoch = epoch + 1
                        logger.info(f"Best model saved at epoch {best_epoch}, dice: {best_dice:.4f}")

                        # save best model
                        best_model_path = os.path.join(dirs["checkpoints_dir"], "best_epoch.pth")

                        torch.save(model.state_dict(), best_model_path)

                        # reset early stopping counter
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
            # early stopping
            if epochs_no_improve >= config['early_stop_patience']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        logger.info(f"Training completed, best_dice: {best_dice:.4f} at epoch: {best_epoch}")
        del model

        # Training completed, now testing
        logger.info(f"Testing started, loading best model at epoch: {best_epoch}")
        model = OurMethod(net_work_config).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        test_labels_list = []
        test_outputs_list = []
        result_dir = os.path.join(dirs["run_dir"], "segmentation")
        os.makedirs(result_dir, exist_ok=True)

        with torch.no_grad():
            for test_data in test_loader:
                ct = test_data["CT"]
                pt = test_data["PET"]
                test_labels = test_data["MASK"]
                test_inputs = torch.cat([ct, pt], dim=1).to(device)

                test_labels = test_labels.to(device)
                test_outputs = sliding_window_inference(test_inputs, (32, 224, 224), 4, model)
                test_outputs = torch.sigmoid(test_outputs)
                test_outputs_list.append(test_outputs)
                test_labels_list.append(test_labels)

                batch_size = ct.shape[0]
                for i in range(batch_size):
                    patient_id = test_data["PatientID"][i]
                    scan_index = test_data["ScanIndex"][i]
                    group_index = test_data["GroupIndex"][i]

                    pred = test_outputs[i]
                    gt = test_labels[i].cpu().numpy()

                    save_path = os.path.join(
                        result_dir, f"{patient_id}_{scan_index}_{group_index}_pred.npy"
                    )
                    gt_save_path = os.path.join(
                        result_dir, f"{patient_id}_{scan_index}_{group_index}_gt.npy"
                    )
                    np.save(save_path, pred.cpu().numpy())
                    np.save(gt_save_path, gt)

                del test_inputs, test_labels, test_outputs
                torch.cuda.empty_cache()

            test_outputs_list = torch.cat(test_outputs_list)
            test_labels_list = torch.cat(test_labels_list)
            test_dice, test_precision, test_recall = compute_metrics(test_outputs_list, test_labels_list)
            logger.info(f"Test Dice Score: {test_dice:.4f}, Precision: {test_precision:.4f}, "
                        f"Recall: {test_recall:.4f}")
            logger.info("Testing at Best Model {}".format(best_model_path))

    except Exception as e:
        logger.info(f"An error occurred during training: {str(e)}")
        raise

    finally:
        del model, train_loader, valid_loader, test_loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)

        train_config['learning_rate'] = float(train_config['learning_rate'])
        train_config['weight_decay'] = float(train_config['weight_decay'])

        run_fold_training(train_config)
