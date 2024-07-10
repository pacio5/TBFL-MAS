import copy
import torch
from tqdm import tqdm
from tqdm.contrib import tzip

class Learning:
    @staticmethod
    def average_gradients(avg, gradients):
        # average gradients
        first_gradient_key = list(gradients.keys())[0]
        avg["gradients"] = copy.deepcopy(gradients[first_gradient_key])
        for i in tqdm(list(gradients.keys())[1:]):
            for j in tqdm(gradients[i].keys()):
                if isinstance(gradients[i][j], dict):
                    for k in gradients[i][j].keys():
                        for l in gradients[i][j][k].keys():
                            avg["gradients"][j][k][l] = torch.add(avg["gradients"][j][k][l], gradients[i][j][k][l])

        for i in tqdm(avg["gradients"].keys()):
            if isinstance(avg["gradients"][i], dict):
                for j in tqdm(avg["gradients"][i].keys()):
                    for k in avg["gradients"][i][j].keys():
                        avg["gradients"][i][j][k] = torch.div(avg["gradients"][i][j][k], len(gradients.keys()))

    @staticmethod
    def average_weights(avg, weights):
        # average weights
        first_weight_key = list(weights.keys())[0]
        avg["weights"] = copy.deepcopy(weights[first_weight_key])
        for i in tqdm(list(weights.keys())[1:]):
            for j in tqdm(weights[i].keys()):
                avg["weights"][j] = torch.add(avg["weights"][j], weights[i][j])
        for i in tqdm(avg["weights"].keys()):
            avg["weights"][i] = torch.div(avg["weights"][i], len(weights.keys()))


    def gradient_descent(criterion, device, model, optimizer, x_train, y_train):
        # setting model up for training
        model.train()
        # make gradient descent step
        for images, labels in tzip(x_train.values(), y_train.values(), desc="training model"):
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = torch.sqrt(criterion(outputs, labels))
            loss.backward()
            optimizer.step()


    def predicting(all_labels, all_predictions, criterion, device, model,
                   testing_losses, x_test, y_original_labels, y_test):
        # setting model up for evaluating
        model.eval()
        # calculate predictions
        with torch.no_grad():
            for images, labels, y_original_labels in tzip(x_test.values(), y_test.values(), y_original_labels.values(),
                                                          desc="predicting with model"):
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
