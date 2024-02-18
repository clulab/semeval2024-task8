import torch
import transformers
import torch.nn as nn
from tqdm import tqdm
import pickle
from utils2 import prepare_dataset
from datasets import load_from_disk
import time
import os
import shutil
import random
import numpy as np
torch.cuda.empty_cache()

import requests


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)  
      torch.cuda.manual_seed_all(seed)  

set_seed(42)

def train(train_loader, model, optimizer, scaler, criterion, model_name, device):
    total_loss = 0
    total_accuracy = 0
    total_length = 0
    # label_names = ["yes", "no", "don't know"]
    model = model.train()
    optimizer.zero_grad()
    the_tqdm = tqdm(
        enumerate(train_loader), total=len(train_loader), position=0, leave=True
    )
    for index, batch in the_tqdm:
        sames = False
        batch_acc = 0
        # get the inputs
        input_input_ids = batch["input input_ids"].to(device)
        input_attention_mask = batch["input attention_mask"].to(device)
        # label_attention_mask = batch["label attention_mask"].to(device)
        label = batch["label"].to(device)
        # label_id = batch["label id"].to(device)
        # label_id[label_id == 3] = 2
        label_id = label
        id_ = batch["id"].to(device)
        assert (not torch.isnan(input_input_ids).any())
        # forward + backward + optimize
        if model_name == "roberta" or model_name == "deberta":
            with torch.cuda.amp.autocast():
                output = model(
                    input_ids=input_input_ids,
                    attention_mask=input_attention_mask,
                    label_ids=label_id,
                ) # .to(device)
                loss = output.loss

            scaler.scale(loss).backward()
            if True: # (index + 1) % 2 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()



        # calculate accuracy
        total_length += len(id_)
        total_loss += loss.item() * len(id_)

        if model_name == "roberta" or model_name == "deberta" or model_name == "bart":
            predicted_label_indices = torch.argmax(output.logits, dim=1)
            # preds = [label_names[idx] for idx in predicted_label_indices.tolist()]
            preds = [idx for idx in predicted_label_indices.tolist()]

        for i in range(len(preds)):
            if preds[i] == label[i]:
                total_accuracy += 1
                batch_acc += 1
        # check if model is predicting the same label for all instances
        if len(set(preds)) == 1:
            sames = True



        overall_loss = total_loss / total_length
        overall_accuracy = total_accuracy / total_length * 100
        batch_acc = batch_acc / len(id_) * 100

        the_tqdm.set_description(f"Loss: {loss.item():.3f}, Acc: {overall_accuracy:.2f}, Batch Acc: {batch_acc:.2f}, Same: {sames}")

    return overall_loss, overall_accuracy


def evaluate(data_loader, model, model_name, device):
    model = model.eval()
    total_loss = 0
    total_accuracy = 0
    total_length = 0
    # label_names = ["yes", "no", "don't know"]
    all_preds = {}
    with torch.no_grad():
        the_tqdm = tqdm(
            enumerate(data_loader), total=len(data_loader), position=0, leave=True
        )
        for index, batch in the_tqdm:
            # get the inputs
            input_input_ids = batch["input input_ids"].to(device)
            input_attention_mask = batch["input attention_mask"].to(device)
            label = batch["label"].to(device)
            # label_attention_mask = batch["label attention_mask"].to(device)
            # label = batch["label"]  # .to(device)
            label_id = label
            # label_id = batch["label id"].to(device)
            # label_id[label_id == 3] = 2
            id_ = batch["id"].to(device)

            # forward + backward + optimize
            output = model(
                input_ids=input_input_ids,
                attention_mask=input_attention_mask,
                # label_input_ids=label_input_ids,
                label_ids=label_id,
            )  # .to(device)
            loss = output.loss

            # calculate accuracy
            total_length += len(id_)
            total_loss += loss.item() * len(id_)

            if model_name == "roberta" or model_name == "deberta" or model_name == "bart":
                predicted_label_indices = torch.argmax(output.logits, dim=1)
                # preds = [label_names[idx] for idx in predicted_label_indices.tolist()]
                preds = [idx for idx in predicted_label_indices.tolist()]

            for i in range(len(preds)):
                if preds[i] == label[i]:
                    total_accuracy += 1

            for i in range(len(id_)):
                all_preds[int(id_[i])] = preds[i]

            overall_loss = total_loss / total_length
            overall_accuracy = total_accuracy / total_length * 100

            the_tqdm.set_description(
                f"Loss: {overall_loss:.3f}, Eval Acc: {overall_accuracy:.2f}"
            )

    return overall_loss, overall_accuracy, all_preds


def getPreds(data_loader, model, model_name, device):
    model = model.eval()
    all_preds = {}
    with torch.no_grad():
        the_tqdm = tqdm(
            enumerate(data_loader), total=len(data_loader), position=0, leave=True
        )
        for index, batch in the_tqdm:
            # get the inputs
            input_input_ids = batch["input input_ids"].to(device)
            input_attention_mask = batch["input attention_mask"].to(device)
            label = batch["label"].to(device)
            label_id = label
            id_ = batch["id"].to(device)

            # forward + backward + optimize
            output = model(
                input_ids=input_input_ids,
                attention_mask=input_attention_mask,
                label_ids=label_id,
            )  # .to(device)

            if model_name == "roberta" or model_name == "deberta" or model_name == "bart":
                predicted_label_indices = torch.argmax(output.logits, dim=1)
                preds = [idx for idx in predicted_label_indices.tolist()]

            for i in range(len(id_)):
                all_preds[int(id_[i])] = preds[i]

    return all_preds


def experiment(
    device,
    model_name="unified",
    setting="qa",
    batch_size_=16,
    experiment_id="test",
    learning_rate=5e-5,
    ids_set=None,
):
    """
    This is the main function to run the experiments.
    """
    # set the hyperparameters
    """
    CondaQA paper:
        BERT:
            learning rate: 1e-5
            epochs: 10
        UnifiedQA:
            learning rate: 5e-5
            epochs: 5
    """
    if model_name == "roberta":
        # learning_rate = 1e-5
        epochs_ = 10
        from models import RoBERTa_negation as Model_QA_negation
        from transformers import RobertaTokenizerFast as Tokenizer
        tokenizer = Tokenizer.from_pretrained("roberta-large")
    elif model_name == "deberta":
        # learning_rate = 1e-5
        epochs_ = 10
        from models import DeBERTa_negation as Model_QA_negation
        from transformers import DebertaV2TokenizerFast as Tokenizer
        tokenizer = Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")

    epoch_num = 0
    best_val_acc = 0
    best_val_loss = 100000
    patience = 2
    check_stopping = 0

    model = Model_QA_negation(device)
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    # load the data
    train_loader, val_loader, test_loader = prepare_dataset(
        model_name=model_name, setting=setting, batch_size1=batch_size_, tokenizer = tokenizer, ids_set=ids_set
    )
    ''''''
    # create the files
    folder_name = "experiments/" + experiment_id + "/"
    data_folder = folder_name + "data/"
    model_folder = folder_name + "model/"
    train_folder = folder_name + "train/"
    code_folder = folder_name + "code/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    '''
    # create a snapshot of the code
    if not os.path.exists(code_folder):
        shutil.copytree("code", code_folder)
    else:
        shutil.rmtree(code_folder)
        shutil.copytree("code", code_folder)
    '''
    save_best_loss = model_folder + "best_loss.pt"
    save_current = model_folder + "current.pt"

    training_info = open(train_folder + "training_info.txt", "w")
    training_info.write(
        f"experiment_id: {experiment_id}\n"
        f"model_name: {model_name}\n"
        f"setting: {setting}\n"
        f"------------------------------------\n"
        f"learning_rate: {learning_rate}\n"
        f"batch_size: {batch_size_}\n"
        f"patience: {patience}\n"
        f"------------------------------------\n"
    )

    training_info.write(f"current time: {time.ctime()}\n")
    start_time = time.time()

    if check_stopping < patience:
        while True:
            epoch_num += 1
            print(f"------------------------------------")
            print(f"Epoch {epoch_num} started...")
            training_info.write(f"Epoch {epoch_num} started...\n")
            print(f"Current patience: is {check_stopping}.")
            training_info.write(f"Current patience: is {check_stopping}.\n")
            # train
            train_loss, train_acc = train(
                train_loader, model, optimizer, scaler, criterion, model_name, device
            )
            print(f"Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.1f}")
            training_info.write(
                f"Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.1f}\n"
            )
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, save_current)

            # evaluate on validation set
            val_loss, val_acc, _ = evaluate(val_loader, model, model_name, device)
            print(f"Val Loss: {val_loss:.2f}, Val Acc: {val_acc:.1f}")
            training_info.write(f"Val Loss: {val_loss:.2f}, Val Acc: {val_acc:.1f}\n")

            # set the early stopping
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                check_stopping = 0
                torch.save(save_dict, save_best_loss)
                print("Validation acc increased, saving the model...")

            else:
                check_stopping += 1

            training_info.write(f"------------------------------------\n")
            if (check_stopping >= patience):
                break
            

    print(f"Training stopped at epoch {epoch_num}.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    training_info.write(f"Training time: {elapsed_time:.2f} seconds.\n")
    # model = Model_QA_negation(device)
    # model = model.to(device)
    # load_dict = torch.load(model_folder + "best_loss.pt")
    # model.load_state_dict(load_dict['model_state_dict'])

    # optimizer.load_state_dict(load_dict['optimizer_state_dict'])
    print(f"Model loaded from the best loss checkpoint.")
    # testPreds = getPreds(test_loader, model, model_name, device)
    
    test_loss, test_acc, preds = evaluate(test_loader, model, model_name, device)
    with open(model_folder + "testPreds.pkl", "wb") as f:
        pickle.dump(preds, f)
    training_info.write(f"Training stopped at epoch {epoch_num}.\n")
    training_info.write(f"Best Val Loss: {best_val_loss:.2f}\n")
    training_info.write(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.1f}\n")
    training_info.close()
    # process_test_preds(preds, data_folder, model_name, setting, best=True)
    print("Training finished Successfully.")

def process_test_preds(preds, data_folder, model_name, setting, best=True):
    """
    This function processes the predictions of the test set.
    """
    if best:
        path = data_folder + "test"
    else:
        path = data_folder + "test2"
    test_set = "data/" + model_name + "/" + setting + "/"
    test_set = load_from_disk(test_set + "test")
    test_set = test_set.add_column("pred", ["DUMMY" for i in range(len(test_set))])
    test_set = test_set.add_column("correct", [-1 for i in range(len(test_set))])
    # test_set = test_set.remove_columns(["tokenized input", "tokenized label"])
    test_set = test_set.map(test_mapper, fn_kwargs={"preds": preds})

    test_set.save_to_disk(path)


def test_mapper(example, preds):
    return_dict = {}
    return_dict["pred"] = preds[example["id"]]
    return_dict["correct"] = 1 if str(example["label"]).strip().lower() == str(preds[example["id"]]).strip().lower() else 0
    return return_dict

from argparse import ArgumentParser
if __name__ == "__main__":
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description="Process some inputs.")
    # Add an argument
    parser.add_argument('input', type=str, help='An input to process')

    # Parse the arguments
    args = parser.parse_args()

    # model_lr
    experiment_id = args.input
    model_, lr = experiment_id.split("_")
    if lr == "1e-5":
        lr = 1e-5
    elif lr == "5e-5":
        lr = 5e-5
    elif lr == "1e-6":
        lr = 1e-6
    elif lr == "5e-6":
        lr = 5e-6
    
    
    try:
        experiment(
            device=device_,
            model_name=model_,
            setting=str(model_) + "_" + str(lr) + "_" + str(lr),
            batch_size_=4, 
            experiment_id=str(model_) + "_" + str(lr) + "_" + str(lr),            
            learning_rate=lr, # float(lr),
            ids_set=None
        )
        notif = (
            f"Training of {model_} with {lr} training data is finished!"
        )
        requests.post(
            "https://ntfy.sh/mhrnlpmodels", data=notif.encode(encoding="utf-8")
        )

        os.system("git add .")
        os.system("git commit -m " + notif)
        os.system("git push")

    except Exception as e:
        errors = open("errors.txt", "a")
        notif = (
            f"Training of {model_} with {lr} training data is finished with error {e}!"
        )
        requests.post(
            "https://ntfy.sh/mhrnlpmodels",
            data=notif.encode(encoding="utf-8"),
            headers={"Priority": "5"},
        )
        errors.write(notif + "\n" + str(e) + "\n")
        errors.write("--------------------------------------------------\n")
        errors.close()
