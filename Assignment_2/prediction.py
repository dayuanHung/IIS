
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, features_in=20, features_out=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 128),
            nn.ReLU(),
            #nn.Linear(512, 128),
            #nn.ReLU(),
            nn.Linear(128, features_out)
        )

    def forward(self, input):
        return self.net(input)


def main():

    data = pd.read_csv("test_to_submit.csv")
    #remove title
    data = data.iloc[:, 0:].values
    data = torch.from_numpy(data).float()
    
    model = MLP(20, 7) 
    model.load_state_dict(torch.load('best_model_lr0.0005.pth'))
    model.eval()

    with torch.no_grad():
        predictions = model(data)
    
    labels = {'neutral': 0, 'disgust': 1, 'sad': 2, 'happy': 3, 'surprise': 4, 'angry': 5, 'fear': 6}
    
    max_indices = np.argmax(predictions, axis=1)
    outputs = [emotion for idx in max_indices for emotion, label in labels.items() if label == idx]
    #out = pd.DataFrame({'Predicted': outputs})
    with open('outputs', 'w') as file:
        file.write('prediction\n')
        for label in outputs:
            file.write(f"{label}\n")

if __name__ == "__main__":
    main()