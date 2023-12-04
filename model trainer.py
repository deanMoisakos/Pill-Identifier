import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

torch.manual_seed(42) #seed for random generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use cpu for the model

data_path = "c:/Users/c000832213/Documents/DeanStuff/snek/Pill Identifier/dataset" #data path for dataset      
classes = ['pill', 'non_pill'] #folder names for classes

#additional transformation for data augmentation
augmentation_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
])

#combine both transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([augmentation_transform, transforms.ToTensor()]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    #transformation for data augmentation
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
    ])

    #transformation for consistent processing
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([augmentation_transform]),
    transforms.ToTensor(),  # Convert PIL image to tensor before normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    #load dataset with transformations applied
    dataset = ImageFolder(root=data_path, transform=transform)

    #split total data set into training and validation
    train_size = int(0.8 * len(dataset)) #80% of data goes to training, rest goes to validation
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    #training and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', weights=None)  #model resnet18 is loaded from pytorch
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  #optimize models parameters during training

# Training loop
num_epochs = 5  #number of training sessions the model will perform
threshold = 0.75  #accuracy threshold for model

for epoch in range(num_epochs):
    model.train()  #begin model training
    train_loss = 0.0  #variables to track models loss, correct, and total
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

        #check if probability of being a pill exceeds threshold
        for i in range(len(predicted)):
            if max_prob[i].item() > threshold:
                pill_label = classes[predicted[i].item()]
            else:
                pill_label = 'Not a Pill'  

        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100.0 * train_correct / train_total  #calculate the training accuracy as a percentage
    avg_train_loss = train_loss / len(train_loader)


    #evaluation on validation set
    model.eval()  #load the validation dataset and evaluate the models performance
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)  #update validation statistics
            val_loss += loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100.0 * val_correct / val_total  #calculate validation accuracy as a percentage
    avg_val_loss = val_loss / len(val_loader)

    #print training statistics
    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

#save the trained model
model_path = os.path.join(data_path, "pill_classifier.pth")
torch.save(model.state_dict(), model_path)
