import os
import argparse
import datetime

import numpy as np
import torch

from dataset.dataset_getter import DatasetGetter
from utils.torch import get_device, save_model, load_model
from utils.log import TensorboardLogger
from utils.config import save_yaml, load_from_yaml
from utils.visdom_monitor import VisdomMonitor
from transformers_for_segmentation.get_model import get_model
from transformers_for_segmentation.base.learner import BaseLearner as Learner


def get_current_time() -> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time


def run(args):
    device = get_device(args.device)

    # Getting Dataset
    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name,
        path=args.dataset_path,
        is_train=not args.test,
        transform=None,
    )
    dataset_loader = DatasetGetter.get_dataset_loader(
        dataset=dataset, batch_size=1 if args.test else args.batch_size
    )
    with torch.no_grad():
        sampled_data = next(iter(dataset_loader))[0]
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
        n_patch=args.patch_size,
        n_dim=args.embedding_size,
        n_encoder_blocks=args.encoder_blocks_num,
        n_heads=args.heads_num,
        use_cnn_embedding=args.use_cnn_embedding,
        n_classes=args.num_classes,
    ).to(device)

    if args.load_from is not None:
        load_model(model, args.load_from)

    # Train / Test Iteration
    learner = Learner(model=model, n_classes=args.num_classes)
    epoch = 1 if args.test else args.epoch

    if args.use_visdom_monitoring:
        visdom = VisdomMonitor()

    if not args.test:
        model_save_dir = "{}/{}/".format(args.save_dir, get_current_time())
        logger = TensorboardLogger(model_save_dir)
        save_yaml(vars(args), model_save_dir + "config.yaml")

    for epoch in range(epoch):
        loss_list, dice_list = [], []
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            learning_info = learner.step(
                images=images, labels=labels, is_train=not args.test
            )
            if args.use_visdom_monitoring:
                visdom.add_train_images(input_batches=images, label_batches=labels)
                visdom.add_batched_label_images(
                    label_batches=learning_info["preds"], caption="Predicted Output"
                )
            loss_list.append(learning_info["loss"])
            dice_list.append(learning_info["dice"])
        loss_avg = np.mean(loss_list)
        dice_avg = np.mean(dice_list)
        print("[Epoch {}] Loss : {} | Dice : {}".format(epoch, loss_avg, dice_avg))
        if not args.test:
            # Save model
            if (epoch + 1) % args.save_interval == 0:
                save_model(model, model_save_dir, "epoch_{}".format(epoch + 1))
            # Log
            logger.log(tag="Training/Loss", value=loss_avg, step=epoch + 1)
            logger.log(tag="Traning/Dice Score", value=dice_avg, step=epoch + 1)
        else:
            break
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
    # model
    parser.add_argument(
        "--model-name", type=str, default="deformable_unetr", help="Model name"
    )
    parser.add_argument("--patch-size", type=int, default=16, help="Image patch size")
    parser.add_argument(
        "--embedding-size", type=int, default=768, help="Number of hidden units"
    )
    parser.add_argument(
        "--encoder-blocks-num",
        type=int,
        default=12,
        help="Number of transformer encoder blocks",
    )
    parser.add_argument(
        "--heads-num", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--use-cnn-embedding",
        action="store_true",
        help="Whether to use cnn based patch embedding",
    )
    # train / test
    parser.add_argument("--epoch", type=int, default=1000, help="Learning epoch")
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
