#################### process dataset ####################
from torch.utils.data import DataLoader
import torch
    
def collateD(batch):
    """
    This function collates the batch.
    Args:
        batch: batch
    Returns:
        batch: collated batch
    """
    input_input_ids = []
    input_attention_mask = []
    label_input_ids = []
    label_attention_mask = []
    label = []
    label_id = []
    id_ = []
    labeltoid = {"yes": 0, "no": 1, "don't know": 2}
    all_texts = []
    for instance in batch:
        
        input_input_ids.append(instance["input_ids"])
        input_attention_mask.append(instance["attention_mask"])
        
        # all_texts.append(instance["input text"])

        # label_input_ids.append(instance["label"])
        # label_attention_mask.append(instance["tokenized label"]["attention_mask"])
        label.append(instance["label"])
        id_.append(instance["id"])
        # label_ = instance["label"].lower().strip()



    input_input_ids = torch.tensor(input_input_ids, dtype=torch.long)
    input_attention_mask = torch.tensor(input_attention_mask, dtype=torch.long)
    # label_input_ids = torch.tensor(label_input_ids, dtype=torch.long)
    # label_attention_mask = torch.tensor(label_attention_mask, dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)
    # label_id = torch.tensor(label_id, dtype=torch.long)
    id_ = torch.tensor(id_, dtype=torch.long)

    return_dict = {
        "input input_ids": input_input_ids,
        "input attention_mask": input_attention_mask,
        # "label input_ids": label_input_ids,
        # "label attention_mask": label_attention_mask,
        "label": label,
        # "label id": label_id,
        "id": id_,
    }

    return return_dict
    

from datasets import load_dataset, load_from_disk
def prepare_dataset(model_name: str, setting: str, batch_size1: int = 16, tokenizer=None, ids_set=None):
    """
    This function prepares the dataset.
    """
    # read the tokenized datasets
    # train = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_train.jsonl', split='train')
    # val = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_dev.jsonl', split='train')
    # test = load_dataset('json', data_files='../datasets/SubtaskB/subtaskB_test.jsonl', split='train')
    train = load_from_disk('../datasets/SubtaskB/train')
    val = load_from_disk('../datasets/SubtaskB/dev')
    test = load_from_disk('../datasets/SubtaskB/test')
    print(train)
    print(val)
    print(test)

    # add id column
    train = train.add_column('id', [i for i in range(len(train))])
    val = val.add_column('id', [i for i in range(len(val))])
    test = test.add_column('id', [i for i in range(len(test))])
    
    # test = test.add_column('label', [100] * len(test))

    # if ids_set is not None:
        # print(type(ids_set))
        # print(ids_set)
        # print(train)
    #     train = train.filter(lambda example: example['id'] in ids_set)
        
    def tokenize_function(examples):
       return tokenizer(examples["paragraph"], truncation=True, padding="max_length", max_length=512)
    
    train = train.map(tokenize_function, batched=True, batch_size=batch_size1)
    val = val.map(tokenize_function, batched=True, batch_size=batch_size1)
    test = test.map(tokenize_function, batched=True, batch_size=batch_size1)

    # train = train[:150]



    # convert the tokenized datasets to torch tensors
    train = DataLoader(
        dataset=train,
        batch_size=batch_size1,
        shuffle=True,
        num_workers=0,
        collate_fn=collateD,
    )
    val = DataLoader(
        dataset=val,
        batch_size=batch_size1,
        shuffle=True,
        num_workers=0,
        collate_fn=collateD,
    )
    
    test = DataLoader(
        dataset=test,
        batch_size=batch_size1,
        shuffle=True,
        num_workers=0,
        collate_fn=collateD,
    )
    

    return train, val, test