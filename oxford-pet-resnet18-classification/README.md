# Oxford-IIIT Pet Breed Classification

A transfer learning project with a pretrained ResNet18 model to classify 37 pet breeds from the Oxford-IIIT Pet dataset.

## Model

- ResNet18 pretrained on ImageNet
- Frozen layers except for the final one
- Replaced final fully connected layer with 37 output classes

## Results

- Best validation loss: 0.3950
- Test loss: 0.4971
- Test accuracy: 84.71%
