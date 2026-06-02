"""WikiText-103 data pipeline with GPT-2 tokenizer and sequence packing."""

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class PackedTextDataset(Dataset):
    """Pre-tokenized, packed sequences of fixed length."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        n_seqs = len(tokens) // (seq_len + 1)
        self.data = tokens[: n_seqs * (seq_len + 1)].view(n_seqs, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {"input_ids": row[:-1], "targets": row[1:]}


class WikiText103DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, seq_len: int = 512, num_workers: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def prepare_data(self):
        load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    def setup(self, stage=None):
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        self.train_ds = PackedTextDataset(self._tokenize_split(ds["train"]), self.seq_len)
        self.val_ds = PackedTextDataset(self._tokenize_split(ds["validation"]), self.seq_len)
        self.test_ds = PackedTextDataset(self._tokenize_split(ds["test"]), self.seq_len)

    def _tokenize_split(self, split) -> torch.Tensor:
        all_ids = []
        for example in split:
            text = example["text"]
            if text.strip():
                all_ids.extend(self.tokenizer.encode(text))
        return torch.tensor(all_ids, dtype=torch.long)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0,
        )
