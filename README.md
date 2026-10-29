# Neural Named Entity Recognition: Feed-Forward NN & LSTM (PyTorch)

**Named Entity Recognition** (tags `PER`, `ORG`, `LOC`, `MISC`, `O`) with neural networks in **PyTorch**, built for the *Natural Language Processing* course, compared against each other and against the earlier Markov-model baselines ([HMM / MEMM repo](https://github.com/omaremad02/Markov-Models-for-Named-Entity-Recognition)).

## Models
### Feed-Forward Neural Network (`NER-Feed-Forward-Neural-Networks.ipynb`)
- A custom `FixedWindowDataset` turns each word plus its surrounding context window into one training example
- Word embeddings → hidden layer → softmax over the 5 tags
- **Average F1 ≈ 0.87 on validation**, a big improvement over the HMM/MEMM baselines

### LSTM (`NER-Long-Short-Term-Memory.ipynb`)
- A custom `SequenceDataset` with **padding and masking**, so batches can mix sentences of different lengths and the loss only counts real tokens
- One-layer LSTM tagger, trained in both **bidirectional** and **unidirectional** versions to measure how much right-hand context helps

## Training & evaluation
Custom PyTorch training and validation loops (`DataLoader`, optimiser, cross-entropy loss), tracking loss, accuracy, per-tag F1 and the confusion matrix every epoch. GPU-ready (trained on Google Colab).

## Tech
Python · PyTorch · scikit-learn · NLP · sequence labelling · LSTM / BiLSTM · word embeddings
