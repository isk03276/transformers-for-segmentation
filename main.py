import os
import argparse
import datetime

import numpy as np
import torch
from torch.utils.data import random_split

from dataset.dataset_getter import DatasetGetter
from utils.torch import get_device, save_model, load_model
from utils.log import TensorboardLogger
from utils.config import save_yaml, load_from_yaml
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


def get_dataset(args):
    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name, path=args.dataset_path, transform=None,
    )
    if args.test or args.validation_set_ratio == 0.0:
        main_dataset = dataset
        validation_dataset = None
    elif args.validation_set_ratio > 0.0 and args.validation_set_ratio < 1.0:
        n_data = len(dataset)
        n_validation_data = int(n_data * args.validation_set_ratio)
        n_train_data = n_data - n_validation_data
        main_dataset, validation_dataset = random_split(
            dataset=dataset, lengths=[n_train_data, n_validation_data]
        )
    else:
        raise Exception
    return main_dataset, validation_dataset


def run_one_epoch(dataset_loader, model_interface, device, visdom=None):
    loss_list, dice_list = [], []
    for images, labels in dataset_loader:
        images = images.to(device)
        labels = labels.to(device)
        result_dict = model_interface.step(
            images=images, labels=labels, is_train=not args.test
        )
        if args.use_visdom_monitoring:
            visdom.add_train_images(input_batches=images, label_batches=labels)
            visdom.add_batched_label_images(
                label_batches=result_dict["preds"], caption="Predicted Output"
            )
        loss_list.append(result_dict["loss"])
        dice_list.append(result_dict["dice"])
    loss_avg = np.mean(loss_list)
    dice_avg = np.mean(dice_list)
    return loss_avg, dice_avg


def run(args):
    device = get_device(args.device)

    # Getting Dataset
    main_dataset, validation_dataset = get_dataset(args)

    # Getting Dataset Loader
    main_dataset_loader = DatasetGetter.get_dataset_loader(
        main_dataset, batch_size=1 if args.test else args.batch_size
    )

    validation_dataset_loader = (
        DatasetGetter.get_dataset_loader(
            validation_dataset, batch_size=1 if args.test else args.batch_size
        )
        if validation_dataset
        else None
    )

    with torch.no_grad():
        sampled_data = next(iter(main_dataset_loader))[0]
    n_channel, n_seq, image_size = sampled_data.size()[1:4]

    # Model Instantiation
    model_cls = get_model(model_name=args.model_name)

    if args.load_from and args.load_model_config:
        dir_path = os.path.dirname(args.load_from)
        config_file_path = dir_path + "/config.yaml"
        config = load_from_yaml(config_file_path)
        args.patch_size = config["patch_size"]
        args.embedding_size = config["embedding_size"]
        args.encoder_blocks_num = config["encoder_blocks_num"]
        args.heads_num = config["heads_num"]
        args.classes_num = config["classes_num"]

    model = model_cls(
        image_size=image_size,
        n_channel=n_channel,
        n_seq=n_seq,
        n_classes=args.num_classes,
        model_config_file_path=args.model_config_file,
    ).to(device)

    if args.load_from is not None:
        load_model(model, args.load_from)

    # Train / Test Iteration
    model_interface = ModelInterface(model=model, n_classes=args.num_classes)
    epoch = 1 if args.test else args.epoch

    visdom_monitor = VisdomMonitor() if args.use_visdom_monitoring else None

    # Init Logger
    if not args.test:
        model_save_dir = "{}/{}/".format(args.save_dir, get_current_time())
        logger = TensorboardLogger(model_save_dir)
        save_yaml(vars(args), model_save_dir + "config.yaml")

    for epoch in range(epoch):
        train_loss_avg, train_dice_avg = run_one_epoch(
            main_dataset_loader, model_interface, device, visdom_monitor
        )
        print(
            "[Epoch {}] Loss : {} | Dice : {}".format(
                epoch, train_loss_avg, train_dice_avg
            )
        )
        if validation_dataset_loader:
            validation_loss_avg, validation_dice_avg = run_one_epoch(
                validation_dataset_loader, model_interface, device, visdom_monitor
            )
            # Log
            logger.log(tag="Validation/Loss", value=validation_loss_avg, step=epoch + 1)
            logger.log(
                tag="Validation/Dice Score", value=validation_dice_avg, step=epoch + 1
            )

        if not args.test:
            # Save model
            if (epoch + 1) % args.save_interval == 0:
                save_model(model, model_save_dir, "epoch_{}".format(epoch + 1))
            # Log
            logger.log(tag="Training/Loss", value=train_loss_avg, step=epoch + 1)
            logger.log(tag="Training/Dice Score", value=train_dice_avg, step=epoch + 1)
        else:
            break

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
        "--num-classes", type=int, default=14, help="Number of the classes"
    )
    parser.add_argument(
        "--validation-set-ratio",
        type=float,
        default=0.1,
        help="Validation dataset ratio. (this value must  be in [0, 1).)",
    )
    # model
    parser.add_argument("--model-name", type=str, default="unetr", help="Model name")
    parser.add_argument(
        "--model-config-file",
        type=str,
        default="configs/unetr/default_unetr.yaml",
        help="Model name",
    )
    # train / test
    parser.add_argument("--epoch", type=int, default=500, help="Learning epoch")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
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
        "--save-interval", type=int, default=5, help="Model save interval"
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    parser.add_argument(
        "--load-model-config",
        action="store_true",
        help="Whether to use the config file of the model to be loaded",
    )

    args = parser.parse_args()
    run(args)
