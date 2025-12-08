import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# ---------------------------------------
#   Map category → label ID
# ---------------------------------------
CATEGORY2ID = {
    "movie_review": 0,
    "online_shopping": 1,
    "local_business_review": 2
}


# ---------------------------------------
#   Multi-domain Dataset
# ---------------------------------------
class MultiDomainDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load .pkl file
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # REQUIRED FIELDS
        self.texts = data["texts"]              # list[str]
        self.categories = data["category"]      # list[str]

        # Convert category string → integer label
        self.labels = [CATEGORY2ID[c] for c in self.categories]

        print(f"Loaded {len(self.texts)} multi-domain samples from {data_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),          # (L,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------
#   Collate Function (tensor only)
# ---------------------------------------
def collate_fn_multi(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ---------------------------------------
#   DataLoader Constructor
# ---------------------------------------
def create_multi_domain_loader(
    data_path: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
):
    dataset = MultiDomainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_multi,   # 🔥 必须加这个
        drop_last=False,
    )


# ---------------------------------------
#   Wrapper: train / val / test loaders
# ---------------------------------------
def prepare_multi_domain_data(
    data_dir: str,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
):

    train_loader = create_multi_domain_loader(
        os.path.join(data_dir, "multi_train.pkl"),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )

    val_loader = create_multi_domain_loader(
        os.path.join(data_dir, "multi_val.pkl"),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    test_loader = create_multi_domain_loader(
        os.path.join(data_dir, "multi_test.pkl"),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
