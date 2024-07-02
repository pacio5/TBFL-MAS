import copy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
from tqdm.contrib import tzip

def average_gradients(avg, gradients):
    # average gradients
    first_gradient_Key = list(gradients.keys())[0]
    avg["gradients"] = copy.deepcopy(gradients[first_gradient_Key])
    for i in tqdm(list(gradients.keys())[1:]):
        for j in tqdm(gradients[i].keys()):
            if isinstance(gradients[i][j], dict):
                for k in gradients[i][j].keys():
                    for l in gradients[i][j][k].keys():
                        avg["gradients"][j][k][l] = torch.add( avg["gradients"][j][k][l], gradients[i][j][k][l])

    for i in tqdm(avg["gradients"].keys()):
        if isinstance(avg["gradients"][i], dict):
            for j in tqdm(avg["gradients"][i].keys()):
                for k in avg["gradients"][i][j].keys():
                    avg["gradients"][i][j][k] = torch.div(avg["gradients"][i][j][k], len(gradients.keys()))


def average_weights(avg, weights):
    # average weights
    first_weight_Key = list(weights.keys())[0]
    avg["weights"] = copy.deepcopy(weights[first_weight_Key])
    for i in tqdm(list(weights.keys())[1:]):
        for j in tqdm(weights[i].keys()):
            avg["weights"][j] = torch.add(avg["weights"][j], weights[i][j])
    for i in tqdm(avg["weights"].keys()):
        avg["weights"][i] = torch.div(avg["weights"][i], len(weights.keys()))


def calculate_metrics(all_labels, all_predictions, all_test_accuracies, all_test_f1_scores,
                            all_test_precisions, all_test_recalls):
    all_test_accuracies.append(accuracy_score(all_labels, all_predictions))
    all_test_f1_scores.append(
        f1_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions), zero_division=0))
    all_test_precisions.append(
        precision_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions), zero_division=0))
    all_test_recalls.append(
        recall_score(all_labels, all_predictions, average="weighted", labels=np.unique(all_predictions), zero_division=0))

def predicting(all_labels, all_predictions, criterion, device, model, testing_losses, x_test,
                      y_original_labels, y_test):
    # setting model up for evaluating
    model.eval()
    # calculate predictions
    with torch.no_grad():
        for images, labels, y_original_labels in tzip(x_test.values(), y_test.values(), y_original_labels.values(),
                                                      desc="predict with model"):
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = torch.sqrt(criterion(outputs, labels))
            testing_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(y_original_labels.tolist())

def training(criterion, device, model, optimizer, training_losses, x_train, y_train):
    # setting model up for training
    model.train()
    # train the model
    for images, labels in tzip(x_train.values(), y_train.values(), desc="training model"):
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = torch.sqrt(criterion(outputs, labels))
        training_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
