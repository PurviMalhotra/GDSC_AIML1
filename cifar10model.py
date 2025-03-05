import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#classes of the dataset CIFAR-10
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#configuration of device
device=torch.device('cpu')
print(f"Using device: {device}")

def get_transforms():
    transform_train=transforms.Compose([        #data augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                                                       ##data normalization
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))     #Scales pixel values from [0, 255] to [0, 1]
    ])
    
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return transform_train, transform_test


#CNN Model
class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Model, self).__init__()
        
        self.features=nn.Sequential(
            # First conv layer: 3 input channels (RGB), 32 output channels/filters(convolutions) created, kernel sixe 3, padding of 1 pixel 
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),                             #normalizes output
            nn.ReLU(inplace=True),                          #ReLu action
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          #extracts max value in each window
            nn.Dropout(0.2),                                #dropuout for regularization
            
            # Second conv layer: 32 input channel (matches the output of the first block), 64 output channels to learn more complex features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # Fully Connected Layer
        self.classifier=nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),                      #dimension size: 64 channels from prev block, 8*8 spacial dimensions after
            nn.BatchNorm1d(512),                             #multiple pooling layers, flattened input=4096 neurons, outpit size=512
            nn.ReLU(inplace=True),                           #Transformation was: conv layer(4096 dimensions)->fc layer->Compressed ver(512 dimensions)
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x=self.features(x)
        x=x.view(x.size(0), -1)                            #flattening to 1D vector
        x=self.classifier(x)
        return x

def prepare_data():
    #Prepare CIFAR-10 training and testing datasets and dataloaders.
    transform_train, transform_test=get_transforms()        #Get transformations   
    #Create training dataset
    trainset=torchvision.datasets.CIFAR10(
        root='./data', train=True,                         #Storage directory,Training data
        download=True, transform=transform_train           #Auto-download,Data preprocessing 
    )
    #Create testing dataset
    testset=torchvision.datasets.CIFAR10(
        root='./data', train=False, 
        download=True, transform=transform_test
    )
    #Create DataLoader for training data
    trainloader=torch.utils.data.DataLoader(
        trainset, batch_size=64, 
        shuffle=True, num_workers=2
    )
    #Create DataLoader for testing data
    testloader=torch.utils.data.DataLoader(
        testset, batch_size=64, 
        shuffle=False, num_workers=2
    )
    
    return trainloader, testloader

def train_model(model, trainloader, testloader, criterion, optimizer, scheduler):
    
    train_losses=[]                                        #initialization
    
    model.train()
    #Training loop: iterate through 20 epochs
    for epoch in range(20):
        running_loss=0.0                                    #loss for this epoch
        correct=0                                           #correct predictions
        total=0                                             #total sample
        
        for inputs, labels in trainloader:
            inputs, labels=inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()                           #Zero the parameter gradients
                                                            #Prevents gradients from accumulating across batches    
            outputs=model(inputs)                           #Forward pass: compute model predictions
            loss=criterion(outputs, labels)                 #Compute the loss between predictions and true labels
            
            loss.backward()                                 #Backward pass: compute gradients
            optimizer.step()
            
            running_loss += loss.item()                     #update running loss
            
            _, predicted=torch.max(outputs.data, 1)
            total += labels.size(0)                         #updating total samples and corect predictions
            correct += (predicted == labels).sum().item()
        
        scheduler.step()                                    #Step the learning rate scheduler
                                                            #Adjusts learning rate based on the defined schedule
        epoch_loss=running_loss/len(trainloader)            #avg loss for entire epoch
        accuracy=100 * correct / total                      #calculate training accuracy
        train_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{20}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses


def evaluate(model, testloader):
  
    model.eval()                                           #Set the model to evaluation mode
                                                           #This disables layers like dropout and batch normalization 
    #initializing variables again                          #to behave consistently during inference
    correct=0
    total=0
    class_correct=list(0. for _ in range(10))              #Initialized with 10 zeros (one for each CIFAR-10 class)
    class_total=list(0. for _ in range(10))
    
    with torch.no_grad():
        #Iterate through test data batches
        for inputs, labels in testloader:
            inputs, labels=inputs.to(device), labels.to(device)
            
            outputs=model(inputs)                         #Forward pass: get model predictions
            _, predicted=torch.max(outputs.data, 1)       #Get the predicted class (highest probability)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze() #Compute per-class correctness
            
            #Update per-class correct predictions and total samples
            for i in range(len(labels)):
                label=labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'\nTest Accuracy: {100 * correct / total:.2f}%')
    
    print('\nPer-class Accuracy:')
    for i in range(10):
        if class_total[i]>0:
            print(f'{CLASSES[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

def vis_predict(model, testloader):
    
    model.eval()
    
    dataiter=iter(testloader)                                  #iterator created for testholder
    images,labels=next(dataiter)
    images,labels=images.to(device),labels.to(device)          #first batch of images and labels
    
    outputs=model(images)                                      #get model predictions
    _, predicted=torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 3))  #Create a figure to display images
    #visualizing first 5 images
    for i in range(5):
        plt.subplot(1, 5, i+1)
        # Prepare image for display
        img=images[i].cpu().numpy().transpose((1, 2, 0))        #Move image back to CPU and convert to numpy
                                                                #Reverse the normalization applied during data preprocessing
        img=img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f'Pred: {CLASSES[predicted[i]]}\nTrue: {CLASSES[labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    trainloader, testloader = prepare_data()                    #data prep
    model=CIFAR10Model(10).to(device)                           #Create CIFAR-10 model with 10 output classes
                                                                #Move model to the specified device 
    
    criterion=nn.CrossEntropyLoss()                            #Loss Function,CrossEntropyLoss is standard for multi-class classification
    optimizer=optim.Adam(model.parameters(), lr=0.001)         #Optimizer;Adam: Adaptive learning rate optimization algorithm

    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)      #learning rate scheduler
    
    print("Starting Training:")                                #training process
    train_losses = train_model(model, trainloader, testloader, criterion, optimizer, scheduler)
    print("\nEvaluating Model:")
    evaluate(model, testloader)                                 #model evaluation
    print("\nVisualizing Predictions:")
    vis_predict(model, testloader)                              #prediction visualization

if __name__ == '__main__':                                      #Ensure the main function runs only when the script is executed directly
    main()



#resources used: 
#https://cs231n.github.io/convolutional-networks/
#https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
#https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/
#https://stackoverflow.com/questions/72262608/steps-for-machine-learning-in-pytorch

    
