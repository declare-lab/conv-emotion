# dialogue_gcn
Pytorch implementation to paper "DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation". 

## Running
You can run the whole process very easily. Take the IEMOCAP corpus for example:

### Step 1: Preprocess.
```bash
./scripts/iemocap.sh preprocess
```

### Step 2: Train.
```bash
./scripts/iemocap.sh train
```

## Performance Comparision

-|Dataset|Weighted F1
:-:|:-:|:-:
Original|IEMOCAP|64.18%
This Implementation|IEMOCAP|64.10%
