from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
# from transformers import get_constant_schedule_with_warmup
import pytorch_warmup as warmup
from transformers.optimization import Adafactor
import re
from typing import Tuple
import pandas as pd
import random
import argparse
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sacrebleu
import wandb
wandb.login()

train_quy_path = "../data/quechua-spanish/train.quy"
train_es_path = "../data/quechua-spanish/train.es"
train_all_path = "../data/quechua-spanish/train_all.tsv"
train_all_quz_path = "../data/quechua-spanish/train_all_quz.tsv"
dev_quy_path = "../data/quechua-spanish/dev.quy"
dev_es_path = "../data/quechua-spanish/dev.es"
model_save_path = "models/nllb-es-quy"
es_quz_quz_path = "../data/quechua-spanish/parallel_data/es-quz/jw300.es-quz.quz"
es_quz_es_path = "../data/quechua-spanish/parallel_data/es-quz/jw300.es-quz.es"
extra_ctp_path = "../data/chatino-spanish/extra.tsv"
extra_cni_path = "../data/ashaninka-spanish/synthetic.tsv"
extra_hch_path = "../data/wixarika-spanish/extra.tsv"
extra_shp_path = "../data/shipibo_konibo-spanish/extra_all.tsv"
extra_grn_path = "../data/guarani-spanish/extra.tsv"
extra_oto_path = "../data/hñähñu-spanish/extra_std.tsv"
extra_nah_path = "../data/nahuatl-spanish/extra.tsv"
extra_tar_path = "../data/raramuri-spanish/extra.tsv"
extra_aym_path = "../data/aymara-spanish/extra.tsv"
synthetic_aym_path = "../data/aymara-spanish/synthetic.tsv"

lang_map = {"Aymara": "aym_Latn",
            "Bribri": "bzd_Latn",
            "Ashaninka": "cni_Latn",
            "Chatino": "ctp_Latn",
            "Guarani": "grn_Latn",
            "Wixarika": "hch_Latn",
            "Nahuatl": "nah_Latn",
            "Otomi": "oto_Latn",
            "Quechua": "quy_Latn",
            "Shipibo-Konibo": "shp_Latn",
            "Raramuri": "tar_Latn"}

data_map = {"Aymara": {"train": {"src": "../data/aymara-spanish/train.es",
                                 "tgt": "../data/aymara-spanish/train.aym"},
                       "dev": {"src": "../data/aymara-spanish/dev.es",
                               "tgt": "../data/aymara-spanish/dev.aym"}},
            "Bribri": {"train": {"src": "../data/bribri-spanish/train.es",
                                 "tgt": "../data/bribri-spanish/train.bzd"},
                       "dev": {"src": "../data/bribri-spanish/dev.es",
                               "tgt": "../data/bribri-spanish/dev.bzd"}},
            "Ashaninka": {"train": {"src": "../data/ashaninka-spanish/train.es",
                                    "tgt": "../data/ashaninka-spanish/train.cni"},
                          "dev": {"src": "../data/ashaninka-spanish/dev.es",
                                  "tgt": "../data/ashaninka-spanish/dev.cni"}},
            "Chatino": {"train": {"src": "../data/chatino-spanish/train.es",
                                  "tgt": "../data/chatino-spanish/train.ctp"},
                        "dev": {"src": "../data/chatino-spanish/dev.es",
                                "tgt": "../data/chatino-spanish/dev.ctp"}},
            "Guarani": {"train": {"src": "../data/guarani-spanish/train.es",
                                  "tgt": "../data/guarani-spanish/train.gn"},
                        "dev": {"src": "../data/guarani-spanish/dev.es",
                                "tgt": "../data/guarani-spanish/dev.gn"}},
            "Wixarika": {"train": {"src": "../data/wixarika-spanish/train.es",
                                   "tgt": "../data/wixarika-spanish/train.hch"},
                         "dev": {"src": "../data/wixarika-spanish/dev.es",
                                 "tgt": "../data/wixarika-spanish/dev.hch"}},
            "Nahuatl": {"train": {"src": "../data/nahuatl-spanish/train.es",
                                  "tgt": "../data/nahuatl-spanish/train.nah"},
                        "dev": {"src": "../data/nahuatl-spanish/dev.es",
                                "tgt": "../data/nahuatl-spanish/dev.nah"}},
            "Otomi": {"train": {"src": "../data/hñähñu-spanish/train.es",
                                "tgt": "../data/hñähñu-spanish/train.oto"},
                      "dev": {"src": "../data/hñähñu-spanish/dev.es",
                              "tgt": "../data/hñähñu-spanish/dev.oto"}},
            "Quechua": {"train": {"src": "../data/quechua-spanish/train.es",
                                  "tgt": "../data/quechua-spanish/train.quy"},
                        "dev": {"src": "../data/quechua-spanish/dev.es",
                                "tgt": "../data/quechua-spanish/dev.quy"}},
            "Shipibo-Konibo": {"train": {"src": "../data/shipibo_konibo-spanish/train.es",
                                         "tgt": "../data/shipibo_konibo-spanish/train.shp"},
                               "dev": {"src": "../data/shipibo_konibo-spanish/dev.es",
                                       "tgt": "../data/shipibo_konibo-spanish/dev.shp"}},
            "Raramuri": {"train": {"src": "../data/raramuri-spanish/train_nontok.es",
                                   "tgt": "../data/raramuri-spanish/train_nontok.tar"},
                         "dev": {"src": "../data/raramuri-spanish/dev.es",
                                 "tgt": "../data/raramuri-spanish/dev.tar"}}}

def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", type=str, required=True,
                        help="The target language.")
    parser.add_argument("-m", "--model", type=str, default="600m")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size during the training.")
    parser.add_argument("-d", "--data", type=str, default="train_all",
                        help="Training data source. `train_all` or `train`.")
    parser.add_argument("-e", "--num_epochs", type=int, default=10,
                        help="Number of epochs to run.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Number of steps to run.")
    parser.add_argument("--max_quy_token_length", type=int, default=30,
                        help="Max token length for Quechua training samples.")
    parser.add_argument("--max_es_token_length", type=int, default=30,
                        help="Max token length for Spanish training samples.")
    parser.add_argument("--max_es_subtok_length", type=int, default=35,
                        help="Max subword token length for Spanish training samples.")
    parser.add_argument("--training_scheme", default="steps",
                        help="Training scheme. `steps` or `epochs`.")
    parser.add_argument("--model_save_path", type=str, required=True,
                        help="Path to save the trained model.")
    parser.add_argument("--wandb_project_name", type=str, default="nllb-es-quy")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--lr_scheduler", default="multiplicativelr")
    parser.add_argument("--multiplicativelr_lambda", type=float, default=0.99)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    args = parser.parse_args()
    assert args.language in list(lang_map.keys())
    return args

def word_tokenize(text: str) -> str:
    # a very naive word tokenizer for languages with English-like orthography
    return re.findall('(\w+|[^\w\s])', text)

def load_data(lang: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data."""
    assert lang in list(lang_map.keys())
    with open(data_map[lang]["train"]["src"], "r") as f:
        src_lines = f.read().split("\n")
    with open(data_map[lang]["train"]["tgt"], "r") as f:
        tgt_lines = f.read().split("\n")
    train_df = pd.DataFrame({"spa_Latn": src_lines,
                             lang_map[lang]: tgt_lines})
    if lang == "Chatino":
        extra_df = pd.read_csv(extra_ctp_path, sep="\t", names=["source", "spa_Latn", "ctp_Latn"])
        extra_df = extra_df[["spa_Latn", "ctp_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Ashaninka":
        extra_df = pd.read_csv(extra_cni_path, sep="\t", names=["source", "spa_Latn", "cni_Latn"])
        extra_df = extra_df[["spa_Latn", "cni_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Wixarika":
        extra_df = pd.read_csv(extra_hch_path, sep="\t", names=["source", "spa_Latn", "hch_Latn"])
        extra_df = extra_df[["spa_Latn", "hch_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Shipibo-Konibo":
        extra_df = pd.read_csv(extra_shp_path, sep="\t", names=["spa_Latn", "shp_Latn"])
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Guarani":
        extra_df = pd.read_csv(extra_grn_path, sep="\t", names=["source", "spa_Latn", "grn_Latn"])
        extra_df = extra_df[["spa_Latn", "grn_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Otomi":
        extra_df = pd.read_csv(extra_oto_path, sep="\t", names=["source", "spa_Latn", "oto_Latn"])
        extra_df = extra_df[["spa_Latn", "oto_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Raramuri":
        extra_df = pd.read_csv(extra_tar_path, sep="\t", names=["source", "spa_Latn", "tar_Latn"])
        extra_df = extra_df[["spa_Latn", "tar_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Nahuatl":
        extra_df = pd.read_csv(extra_nah_path, sep="\t", names=["source", "spa_Latn", "nah_Latn", "extra"])
        extra_df = extra_df[extra_df.extra.isna()]
        extra_df = extra_df[["spa_Latn", "nah_Latn"]]
        train_df = pd.concat([train_df, extra_df])
    elif lang == "Aymara":
        extra_df = pd.read_csv(extra_aym_path, sep="\t", names=["source", "spa_Latn", "aym_Latn", "extra"])
        extra_df = extra_df[extra_df.extra.isna()]
        syn_df = pd.read_csv(synthetic_aym_path, sep="\t", names=["source", "spa_Latn", "aym_Latn", "extra"])
        syn_df = syn_df[syn_df.extra.isna()]
        extra_df = extra_df[["spa_Latn", "aym_Latn"]]
        syn_df = syn_df[["spa_Latn", "aym_Latn"]]
        train_df = pd.concat([train_df, extra_df, syn_df])
        
    # dev
    with open(data_map[lang]["dev"]["src"], "r") as f:
        src_lines = f.read().split("\n")
    with open(data_map[lang]["dev"]["tgt"], "r") as f:
        tgt_lines = f.read().split("\n")
    dev_df = pd.DataFrame({"spa_Latn": src_lines,
                           lang_map[lang]: tgt_lines})
    return train_df, dev_df

def load_all_train() -> pd.DataFrame:
    assert args.data == "train_all"
    df = pd.read_csv(train_all_path, sep="\t", index_col=0)
    df = df[["es", "quy"]]
    return df

def load_dev() -> pd.DataFrame:
    with open(dev_quy_path, "r") as f:
        dev_quy_lines = f.read().split("\n")
    with open(dev_es_path, "r") as f:
        dev_es_lines = f.read().split("\n")
    dev_df = pd.DataFrame({"es": dev_es_lines,
                           "quy": dev_quy_lines})
    return dev_df

def clean_data(src_lines: list, tgt_lines: list) -> Tuple[list, list]:
    """Clean the data."""
    src_cleaned = []
    tgt_cleaned = []
    for i in range(len(quy_lines)):
        if src_lines[i] in {"", "*"} or \
           tgt_lines[i] in {"", "*"} or \
           src_lines[i] == tgt_lines[i]:
            continue
        # remove non-alphabetical strings
        if args.language == "quy":
            src_line = re.sub(r"[^\w\s]", "", src_lines[i])
            tgt_line = re.sub(r"[^\w\s]", "", tgt_lines[i])
        # remove number digits
        src_line = re.sub(r"\d*", "", src_lines[i])
        tgt_line = re.sub(r"\d*", "", tgt_lines[i])
        src_cleaned.append(src_line)
        tgt_cleaned.append(tgt_line)
    assert len(src_cleaned) == len(tgt_cleaned)
    return src_cleaned, tgt_cleaned

def get_sent_length(text: str) -> int:
    try:
        length = text.split()
    except AttributeError as e:
        print(e)
        print(text)
    return len(text.split())

def filter_by_sent_length(df: pd.DataFrame, lang: str, length=5):
    """Filter by the number of tokens (not subword-tokenized but
    simply splitting by whitespace)
    `lang` (str): Languge name (e.g., "Qeuchua")"""
    assert not df.isnull().values.any()
    assert lang in list(lang_map.keys()) + ["Spanish"]
    if lang == "Spanish":
        df["spa_Latn_length"] = df["spa_Latn"].apply(get_sent_length)
        return df[df["spa_Latn_length"] <= length]
    else:
        col_name = lang_map[lang]
        new_col_name = col_name + "_length"
        df[new_col_name] = df[col_name].apply(get_sent_length)
        return df[df[new_col_name] <= length]

def sort_and_filter(df: pd.DataFrame, lang_code: str):
    """Sort by (subword-tokenized) length and filter out samples
    that are too long.
    `lang_code` (str): e.g., quy_Latn"""
    df = df[~((df["spa_Latn"] == "") |
              (df[lang_code] == ""))]
    df["spa_Latn_tok"] = df["spa_Latn"].apply(tokenizer.tokenize)
    df["spa_Latn_tok_length"] = df["spa_Latn"].apply(len)
    df = df[df["spa_Latn_tok_length"] <= args.max_es_subtok_length]
    df = df.sort_values(by=["spa_Latn_tok_length"])
    return df

def fix_tokenizer(tokenizer, new_lang: str) -> None:
    """Fix the tokenizer to include the new language code.
    args:
    - new_lang (str): the language code of the new language, e.g., `cni_Latn`
    """
    new_special_tokens = tokenizer.additional_special_tokens + [new_lang]
    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
    tokenizer.lang_code_to_id[new_lang] = len(tokenizer) - 1

def get_random_batch_pairs(batch_size: int, data: pd.DataFrame) -> tuple:
    """Randomly sample for a training batch."""
    (l1, lang1), (l2, lang2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(item[l1])
        yy.append(item[l2])
    return xx, yy, lang1, lang2

def get_batch_pairs(batch_size: int, step: int, df: pd.DataFrame, src_lang: str, tgt_lang) -> tuple:
    """Get batch pairs without random sampling.
    Make sure that the training data is sorted by length.
    args:
    - src_lang (str): Source language code (usually spa_Latn)
    - tgt_lang (str): Target language code
    """
    xx, yy = [], []
    for i in range(batch_size):
        if batch_size * step + i >= len(data):
            break
        item = df.iloc[batch_size * step + i]
        xx.append(item[src_lang])
        yy.append(item[tgt_lang])
    return xx, yy

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_lang: str, tgt_lang: str):
        self.df = df
        self.src = df[src_lang]
        self.tgt = df[tgt_lang]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.src.iloc[idx]
        y = self.tgt.iloc[idx]
        return x, y

def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()

def translate(text, lang_code: str, a=32, b=3, max_input_length=1024, num_beams=4, **kwargs):
    tokenizer.src_lang = "spa_Latn"
    tokenizer.tgt_lang = lang_code
    inputs = tokenizer(text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=max_input_length)
    result = model.generate(**inputs.to(device),
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code),
                            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
                            num_beams=num_beams,
                            **kwargs)
    return tokenizer.batch_decode(result, skip_special_tokens=True)

chrf_calc = sacrebleu.CHRF(word_order=2)
def dev_eval(df: pd.DataFrame, lang_code: str) -> sacrebleu.metrics.chrf.CHRFScore:
    """Evaluate the model.
    args:
    - df (pd.DataFrame): data frame of the development set (dev_df)."""
    df["trans"] = [translate(t, lang_code)[0].capitalize() for t in df["spa_Latn"]]
    chrf_result = chrf_calc.corpus_score(dev_df["trans"].tolist(), [dev_df[lang_code].tolist()])
    return chrf_result

# hyperparameters
MAX_LENGTH = 128

model_map = {"600m": "facebook/nllb-200-distilled-600M",
             "1.3b": "facebook/nllb-200-1.3B",
             "3.3b": "facebook/nllb-200-3.3B"}

if __name__ == "__main__":
    args = get_args()
    tokenizer = NllbTokenizer.from_pretrained(model_map[args.model])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_map[args.model])
    if args.data == "train":
        train_df, dev_df = load_data(args.language)
    elif args.data == "train_all" and args.language == "Quechua":
        train_df = load_all_train()
        print(train_df.head())
        print(len(train_df))
        train_df = train_df.dropna(axis=0)
        dev_df = load_dev()
    elif args.data == "train_all_quz" and args.language == "Quechua":
        train_df = pd.read_csv(train_all_quz_path, sep="\t", index_col=0)
        train_df = train_df[["es", "quy"]]
        train_df = train_df.dropna(axis=0)
        dev_df = load_dev()
    print("Dataset size:", len(train_df))
    print(train_df.head())

    lang_code = lang_map[args.language]
    train_df = sort_and_filter(train_df, lang_code)
    print("the size of train_df", len(train_df))
    print(train_df.head())

    # Create the Dataloader
    dataloader = DataLoader(dataset=CustomDataset(train_df, src_lang="spa_Latn", tgt_lang=lang_code),
                            batch_size=args.batch_size,
                            shuffle=False)

    # Add the special token (lang code) to the tokenizer if necessary
    if lang_code not in tokenizer.additional_special_tokens:
        fix_tokenizer(tokenizer, lang_code)
    
    # Training loop
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
        )

    losses = []
    if args.lr_scheduler == "multiplicativelr":
        lmbda = lambda epoch: args.multiplicativelr_lambda
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    else:
        raise NotImplementedError
    if args.warmup:
        warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_steps)
    else:
        raise NotImplementedError

    # Training
    if args.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # use multi-GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run = wandb.init(
        project=args.wandb_project_name,
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs
        }
    )
    
    x, y, loss = None, None, None
    if args.training_scheme == "epochs":
        run = wandb.init(project=args.wandb_project_name,
                         config={"learning_rate": args.learning_rate,
                                 "epochs": args.num_epochs})
        step = 0
        num_ooe = 0
        best_chrf_score = -float("inf")
        model.train()
        for i in range(args.num_epochs):
            step_per_epoch = 0
            for xx, yy in dataloader:
                try:
                    tokenizer.src_lang = "spa_Latn"
                    x = tokenizer(xx, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
                    tokenizer.src_lang = lang_code
                    y = tokenizer(yy, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
                    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                    print("input size:", x.input_ids.size())
                    output = model(**x, labels=y.input_ids)
                    print("output size:", output.logits.size())
                    loss = output.loss
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # warm up; otherwise no change in lr
                    with warmup_scheduler.dampening():
                        pass
                    # Log
                    run.log({"loss": loss,
                             "learning_rate": optimizer.param_groups[0]['lr'],
                             "epoch": i})
                    step_per_epoch += 1
                    step += 1
                except ValueError as e:
                    optimizer.zero_grad(set_to_none=True)
                    x, y, loss = None, None, None
                    cleanup()
                    print("error", e)
                    step_per_epoch += 1
                    step += 1
                    continue
                except TypeError as e:
                    optimizer.zero_grad(set_to_none=True)
                    x, y, loss = None, None, None
                    cleanup()
                    print("error", e)
                    step_per_epoch += 1
                    step += 1
                    continue
                except RuntimeError as e:
                    num_ooe += 1
                    optimizer.zero_grad(set_to_none=True)
                    x, y, loss = None, None, None
                    cleanup()
                    print('error', max(len(s) for s in xx + yy), e)
                    step_per_epoch += 1
                    step += 1
                    continue
                
                if step % 1000 == 0:
                    print(i, np.mean(losses[-1000:]))

            # The end of an epoch; slightly decrease the learning rate
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 > args.warmup_steps:
                    scheduler.step()
            
            # Evaluate
            model.eval()
            chrf_result = dev_eval(dev_df, lang_code)
            print(f"Epoch {i}: {str(chrf_result)}")
            run.log({"eval-chrf": chrf_result.score})
            if best_chrf_score <= chrf_result.score:
                best_chrf_score = chrf_result.score
                model.save_pretrained(args.model_save_path)
                tokenizer.save_pretrained(args.model_save_path)
                with open(f"{args.model_save_path}/stats.txt", "w") as f:
                    f.write(str(chrf_result))
                    f.write("\n")
                    translations = list(df.trans)
                    for t in translations:
                        f.write(t)
                        f.write("\n")
            else:
                if args.early_stopping:
                    print("ChrF score worsened. Early-stopping.")
                    break
