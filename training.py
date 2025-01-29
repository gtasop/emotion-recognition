import torch
import statistics


def calculate_accuracy(pred, labels, original_classes):
    if original_classes:
        TP = torch.sum((pred == 1.0) & (labels == 1.0)).item()
        TN = torch.sum((pred == 0.0) & (labels == 0.0)).item()
        FP = torch.sum((pred == 1.0) & (labels == 0.0)).item()
        FN = torch.sum((pred == 0.0) & (labels == 1.0)).item()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
    else:
        batch_size = labels.shape[0]
        _, pred = torch.max(pred, dim=1)
        correct_predictions = (pred == labels).sum().item()

        accuracy = correct_predictions / batch_size

    return accuracy



def training_img(num_epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_classes,
                 scheduler=None, original_classes=True):
    avg_acc = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            if num_classes == 5:
                labels = labels.long()
            optimizer.zero_grad()
            outputs = model(data)
            if num_classes == 1:
                outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

            predictions = (outputs >= 0.5).float()
            train_accuracy += calculate_accuracy(predictions, labels, original_classes) * data.size(0)
        train_loss /= len(train_loader.dataset)
        train_accuracy /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0


        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                if num_classes == 5:
                    labels = labels.long()
                outputs = model(data)
                if num_classes == 1:
                    outputs = outputs.reshape(-1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                predictions = (outputs >= 0.5).float()
                val_accuracy += calculate_accuracy(predictions, labels, original_classes) * data.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy /= len(val_loader.dataset)

        if scheduler is not None:
            scheduler.step(val_loss)
            print(f"Learning rate: {scheduler.get_last_lr()}")

        avg_acc.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    print(f"Average validation accuracy: {(sum(avg_acc)/len(avg_acc)):.4f}")

    # Testing loop (optional)
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            if num_classes == 5:
                labels = labels.long()
            outputs = model(data)
            if num_classes == 1:
                outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * data.size(0)
            predictions = (outputs >= 0.5).float()
            test_accuracy += calculate_accuracy(predictions, labels, original_classes) * data.size(0)

        test_accuracy /= len(test_loader.dataset)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def training(num_epochs, model, train_loaders, val_loaders, test_loaders, optimizer, criterion, device, num_classes,
             scheduler=None, original_classes=True):
    avg_acc = []
    higest_acc = 0.0
    lowest_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = []
        train_loss = 0.0
        train_acc_total = []
        train_accuracy = 0.0
        for train_loader in train_loaders:
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                if num_classes == 1:
                    outputs = outputs.reshape(-1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

                predictions = (outputs >= 0.5).float()
                train_accuracy += calculate_accuracy(predictions, labels, original_classes) * data.size(0)
            train_loss /= len(train_loader.dataset)
            train_loss_total.append(train_loss)
            train_loss = 0.0
            train_accuracy /= len(train_loader.dataset)
            train_acc_total.append(train_accuracy)
            train_accuracy = 0.0


        model.eval()
        val_loss_total = []
        val_loss = 0.0
        val_acc_total = []
        val_accuracy = 0.0


        with torch.no_grad():
            for val_loader in val_loaders:
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    if num_classes == 1:
                        outputs = outputs.reshape(-1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * data.size(0)
                    predictions = (outputs >= 0.5).float()
                    val_accuracy += calculate_accuracy(predictions, labels, original_classes) * data.size(0)

                val_loss /= len(val_loader.dataset)
                val_accuracy /= len(val_loader.dataset)
                avg_acc.append(val_accuracy)
                if epoch == 0:
                    lowest_acc = val_accuracy
                if val_accuracy < lowest_acc:
                    lowest_acc = val_accuracy
                if val_accuracy > higest_acc:
                    higest_acc = val_accuracy

                val_loss_total.append(val_loss)
                val_acc_total.append(val_accuracy)
                val_loss = 0.0
                val_accuracy = 0.0

            if scheduler is not None:
                scheduler.step(val_loss)
                print(f"Learning rate: {scheduler.get_last_lr()}")

            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {statistics.mean(train_loss_total):.4f}, Train Accuracy: {statistics.mean(train_acc_total):.4f}, "
                  f"Val Loss: {statistics.mean(val_loss_total):.4f}, Val Accuracy: {statistics.mean(val_acc_total):.4f}")

    print(f"Average validation accuracy: {(statistics.mean(avg_acc)):.4f}")
    print(f"Highest validation accuracy: {higest_acc:.4f}")
    print(f"Lowest validation accuracy: {lowest_acc:.4f}")

    # Testing loop (optional)
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    test_accuracy_total = []

    with torch.no_grad():
        for test_loader in test_loaders:
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                if num_classes == 1:
                    outputs = outputs.reshape(-1)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * data.size(0)
                predictions = (outputs >= 0.5).float()
                test_accuracy += calculate_accuracy(predictions, labels, original_classes) * data.size(0)

            test_accuracy /= len(test_loader.dataset)
            test_accuracy_total.append(test_accuracy)

        test_accuracy = statistics.mean(test_accuracy_total)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
