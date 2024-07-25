import os

import click
import torch
import yaml
from transformers import BertTokenizerFast

from model_pytorch_transformers import BERTModel
from predict import predict
from train import evaluate_model, train
from utils import get_books_dataloader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {DEVICE} device")
torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")


@click.group()
def cli():
    pass


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model(tokenizer, config, saved_model_file):
    model = BERTModel(tokenizer.vocab_size, config, DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6}M params")
    model = torch.compile(model)
    if os.path.exists(saved_model_file):
        print(f"Loading existing model: {saved_model_file}")
        model.load_state_dict(
            torch.load(
                saved_model_file, map_location=torch.device(DEVICE), weights_only=True
            )
        )
    return model.to(DEVICE)


# LR grid search
LEARNING_RATE_SEARCH = [10**r for r in range(-5, 0)]


@click.command(name="lr_grid_search")
@click.option("--model_config", default="model_config.yaml")
def lr_grid_search(model_config):
    config = load_config(model_config)
    config["epochs"] = 10
    dataloader = get_books_dataloader(config["batch_size"], max_size=1000)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    scores = {}
    for lr in LEARNING_RATE_SEARCH:
        print(f"Evaluating LR {lr}")
        config["learning_rate"] = lr
        model = get_model(tokenizer, config, None)
        train(
            model,
            dataloader,
            tokenizer,
            config,
            DEVICE,
            None,
        )
        scores[lr] = evaluate_model(model, tokenizer, dataloader, config, DEVICE)
    print(f"LR Scores: {scores}")
    best_lr = min(scores.items(), key=lambda x: x[1])
    print(f"Best LR: {best_lr}")


@click.command(name="train_model")
@click.option(
    "--saved_model_file", "saved_model_file", default="BERT_prepositions/model.pt"
)
@click.option("--model_config", default="model_config.yaml")
def train_model(saved_model_file, model_config):
    config = load_config(model_config)
    dataloader = get_books_dataloader(config["batch_size"])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = get_model(tokenizer, config, saved_model_file)
    train(
        model,
        dataloader,
        tokenizer,
        config,
        DEVICE,
        saved_model_file,
    )


@click.command(name="make_inference")
@click.option("--model", "saved_model_file", default="BERT_prepositions/model.pt")
@click.option("--model_config", default="model_config.yaml")
def make_inference(
    saved_model_file: str,
    model_config,
):
    config = load_config(model_config)
    dataloader = get_books_dataloader(config["batch_size"])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if not os.path.exists(saved_model_file):
        raise Exception("Must have a pretrained model")
    model = get_model(tokenizer, config, saved_model_file)
    predict(model, next(iter(dataloader)), tokenizer, config["context_length"], DEVICE)


cli.add_command(lr_grid_search)
cli.add_command(train_model)
cli.add_command(make_inference)

if __name__ == "__main__":
    cli()
