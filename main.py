import argparse
import datetime

import numpy as np
import torch

from dataset.dataset_managers import DatasetGetter, KFoldManager
from utils.torch import get_device, save_model, load_model
from utils.log import TensorboardLogger
from utils.config import save_yaml
from utils.visdom_monitor import VisdomMonitor
from transformers_for_segmentation.get_model import get_model
from transformers_for_segmentation.common.model_interface import ModelInterface


def get_current_time() -> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time


def run_one_epoch(
    dataset_loader, model_interface, device, is_train, visdom_monitor=None
):
    loss_list, dice_list = [], []
    for images, labels in dataset_loader:
        images = images.to(device)
        labels = labels.to(device)
        result_dict = model_interface.step(
            images=images, labels=labels, is_train=is_train
        )
        if visdom_monitor:
            visdom_monitor.add_train_images(input_batches=images, label_batches=labels)
            visdom_monitor.add_batched_label_images(
                label_batches=result_dict["preds"], caption="Predicted Output"
            )
        loss_list.append(result_dict["loss"])
        dice_list.append(result_dict["dice"])
    return loss_list, dice_list


def train(
    model_interface, k_fold_manager, epoch, dataset_loader, device, visdom_monitor
):
    splits = list(k_fold_manager.split_dataset())
    for epoch in range(epoch // len(splits)):
        for (train_idx, val_idx) in splits:
            results = {
                "Train/Loss": None,
                "Train/Dice Score": None,
                "Validation/Loss": None,
                "Validation/Dice Score": None,
                "Test/Loss": None,
                "Test/Dice Score": None,
            }
            # Train
            k_fold_manager.set_dataset_fold(train_idx)
            train_loss, train_dice = run_one_epoch(
                dataset_loader, model_interface, device, True, visdom_monitor
            )
            results["Train/Loss"] = train_loss
            results["Train/Dice Score"] = train_dice
            # Validation
            k_fold_manager.set_dataset_fold(val_idx)
            val_loss, val_dice = run_one_epoch(
                dataset_loader, model_interface, device, False, visdom_monitor
            )
            results["Validation/Loss"] = val_loss
            results["Validation/Dice Score"] = val_dice
            # Test
            dataset_loader.dataset.set_test_mode()
            test_loss, test_dice = run_one_epoch(
                dataset_loader, model_interface, device, False, visdom_monitor
            )
            results["Test/Loss"] = test_loss
            results["Test/Dice Score"] = test_dice
            yield results


def test(model_interface, dataset_loader, device, visdom_monitor):
    results = {
        "Test/Loss": [],
        "Test/Dice Score": [],
    }
    test_loss, test_dice = run_one_epoch(
        dataset_loader, model_interface, device, False, visdom_monitor
    )
    results["Test/Loss"].extend(test_loss)
    results["Test/Dice Score"].extend(test_dice)
    yield results


def run(args):
    device = get_device(args.device)

    # Getting Dataset
    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name,
        path=args.dataset_path,
        transform=None,
        testset_ratio=args.testset_ratio if not args.test else None,
    )
    n_classes = dataset.n_classes

    # Getting Dataset Loader
    dataset_loader = DatasetGetter.get_dataset_loader(
        dataset, batch_size=1 if args.test else args.batch_size
    )

    # Cross Validation
    k_fold_manager = KFoldManager(dataset, args.n_folds) if not args.test else None

    with torch.no_grad():
        sampled_data = next(iter(dataset_loader))[0]
    n_channel, n_seq, image_size = sampled_data.size()[1:4]

    # Model Instantiation
    model_cls = get_model(model_name=args.model_name)
    model_args = dict(
        image_size=image_size,
        n_channel=n_channel,
        n_seq=n_seq,
        n_classes=n_classes,
    )
    if args.model_config_file:
        model_args["model_config_file_path"] = args.model_config_file
    model = model_cls(**model_args).to(device)
    if args.load_from:
        load_model(model, args.load_from, keywards_to_exclude="decoder_output" if not args.test and args.pretrain else None)

    # Train / Test Iteration
    model_interface = ModelInterface(model=model, n_classes=n_classes)
    epoch = 1 if args.test else args.epoch

    # Init Logger
    visdom_monitor = VisdomMonitor() if args.use_visdom_monitoring else None
    if not args.test:
        model_save_dir = "{}/{}/{}/".format(
            args.save_dir, args.dataset_name, get_current_time()
        )
        logger = TensorboardLogger(model_save_dir)
        save_yaml(vars(args), model_save_dir + "config.yaml")
        save_yaml(model.configs, model_save_dir + "model_config.yaml")

    results = (
        test(model_interface, dataset_loader, device, visdom_monitor)
        if args.test
        else train(
            model_interface,
            k_fold_manager,
            epoch,
            dataset_loader,
            device,
            visdom_monitor,
        )
    )
    for epoch, result in enumerate(results):
        for key, value in result.items():
            if args.test:
                print("[{}] : {}".format(key, np.mean(value)))
            else:
                logger.log(tag=key, value=np.mean(value), step=epoch + 1)
                # Save model
                if (epoch + 1) % args.save_interval == 0:
                    save_model(model, model_save_dir, "epoch_{}".format(epoch + 1))

    if not args.test:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transformer based Networks for Image Segmentation"
    )
    # dataset
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name to use GPU (ex. cpu, cuda, mps, etc.)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="btcv", help="Dataset name (ex. cifar10"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="data/btcv", help="Dataset path"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=0,
        help="Nuber of the folds in the k-fold cross validation(If this value is less than 1, do not cross-validation.",
    )
    parser.add_argument(
        "--testset-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing.",
    )
    # model
    parser.add_argument("--model-name", type=str, default="unetr", help="Model name")
    parser.add_argument(
        "--model-config-file", type=str, help="Model config file path",
    )
    # train / test
    parser.add_argument("--epoch", type=int, default=800, help="Learning epoch")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--pretrain", action="store_true", help="Whether to use the pretrained model('load-from' arg must be activated)")
    parser.add_argument(
        "--use-visdom-monitoring",
        action="store_true",
        help="Whether to visualize inferenced results",
    )
    # save / load
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/", help="Dataset name (ex. cifar10"
    )
    parser.add_argument(
        "--save-interval", type=int, default=50, help="Model save interval"
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    parser.add_argument(
        "--load-model-config",
        action="store_true",
        help="Whether to use the config file of the model to be loaded",
    )

    args = parser.parse_args()
    run(args)
