import os
import nni
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


def evaluate(model_cls):
    net = model_cls()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)

    trainloader = torch.utils.data.DataLoader(train, batch_size = 128, shuffle = True, num_workers = 2)

    test = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(test, batch_size = 128, shuffle = False, num_workers = 2)

    # Initialize Model
    inp = next(enumerate(trainloader))[1]
    net(inp[0])

    # visualizing
    onnx_dir = os.path.abspath(os.environ.get('NNI_OUTPUT_DIR', '.'))
    os.makedirs(onnx_dir, exist_ok = True)
    torch.onnx.export(net, inp[0], onnx_dir + '/model.onnx')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs, labels
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

    print('Training Done')

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images, labels
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print('Accuracy on 10 000 test images: ', 100 * accuracy, '%')
    nni.report_final_result(accuracy)
