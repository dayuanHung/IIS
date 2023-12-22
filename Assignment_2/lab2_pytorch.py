import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import matplotlib.pyplot as plt

#SGD - > adam

class MLP(nn.Module):
    def __init__(self, features_in=20, features_out=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(128, features_out)
        )

    def forward(self, input):
        return self.net(input)


class MultiEmoVA(Dataset):
    def __init__(self, data_path):
        super().__init__()

        data = pd.read_csv(data_path)
        # everything in pytorch needs to be a tensor
        self.inputs = torch.tensor(data.drop("emotion", axis=1).to_numpy(dtype=np.float32))

        # we need to transform label (str) to a number. In sklearn, this is done internally
        self.index2label = [label for label in data["emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}
        

        self.labels = torch.tensor(data["emotion"].apply(lambda x: torch.tensor(label2index[x])))

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)

def plot_loss(losses, val_acc):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.ylim(0.8, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.ylim(25, 70)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.show()
    plt.savefig("MLP1_e20_Adam.png")
    
def plot_lr(LR, lrs_acc):
    LR = [str(num) for num in LR]
    plt.bar(LR, lrs_acc)
    for i, (xi, ui) in enumerate(zip(LR, lrs_acc)):
        plt.text(xi, ui, f'({xi}, {ui:.2f})', ha='center', va='bottom')
    plt.ylim(40, 75)
    plt.xlabel('learning rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.savefig("lr.png")

def main():
    dataset = MultiEmoVA("dataset.csv")

    # passing a generator to random_split is similar to specifying the seed in sklearn
    generator = torch.Generator().manual_seed(18)

    # this can also generate multiple sets at the same time with e.g. [0.7, 0.2, 0.1]
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(  # this loads the data that we need dynamically
        train,
        batch_size=4,  # instead of taking 1 data point at a time we can take more, making our training faster and more stable
        shuffle=True  # Shuffles the data between epochs (see below)
    )

    val_loader = DataLoader(  # this loads the data that we need dynamically
        val,
        batch_size=4,  # instead of taking 1 data point at a time we can take more, making our training faster and more stable
    )

    #print(train[0][0].shape[0], len(dataset.index2label))
    
    #LR = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    LR = [0.001]
    lrs_acc = []
    for l in LR:
        model = MLP(train[0][0].shape[0], len(dataset.index2label))

        optim = torch.optim.Adam(model.parameters(), lr=l)

        loss_fn = nn.CrossEntropyLoss()

        # Check if we have GPU acceleration, if we do our code will run faster
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # if you are on mac with the new M1, M2, ... chips you can try the following instead of cuda
        #device = "mps" if torch.backends.mps.is_available() else device

        print(f"Using device: {device}")

        # we need to move our model to the correct device
        model = model.to(device)
        
        losses = []
        val_acc = []
        best_acc = 0.0
        epochs = 20
        # it is common to do a training loop multiple times, we call these 'epochs'
        for epoch in range(epochs):
            model.train()
            epoch_losses = []
            for inputs, labels in train_loader:
                # both input, output and model need to be on the same device
                inputs = inputs.to(device)
                labels = labels.to(device)

                out = model(inputs)
                loss = loss_fn(out, labels)
                epoch_losses.append(loss.item())

                loss.backward()
                optim.step()
                optim.zero_grad()

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(epoch_loss) 

            model.eval()
            correct = 0
            with torch.no_grad():
                for val_input, val_label in val_loader:
                    val_input = val_input.to(device)
                    val_label = val_label.to(device)
                    val_out = model(val_input)
                    _, predicted = torch.max(val_out, 1)
                    correct += (predicted == val_label).sum().item()

                val_accuracy = correct / len(val) * 100
                val_acc.append(val_accuracy)

                # print(f"Epoch [{epoch + 1}/{epochs}], "
                #     f"Train Loss: {losses[-1]:.4f}, "
                #     f"Val Accuracy: {val_accuracy:.2f}%")

                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    #torch.save(model.state_dict(), 'best_model_lr'+str(l)+'.pth') 
            
        #plot_loss(losses, val_acc)

        
        best_model = MLP(train[0][0].shape[0], len(dataset.index2label)) 
        #best_model.load_state_dict(torch.load('best_model_lr'+str(l)+'.pth'))
        best_model.eval()
    
        with torch.no_grad():
            test_loader = DataLoader(test, batch_size=4)
            correct = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                predictions = model(inputs)

                # Here we go from the models output to a single class and compare to ground truth
                correct += (predictions.softmax(dim=1).argmax(dim=1) == labels).sum()
            print(f"Accuracy is: {correct / len(test) * 100}%")
            
            lrs_acc.append(correct.cpu().numpy() / len(test) * 100)
    #plot_lr(LR, lrs_acc)


if __name__ == "__main__":
    main()
