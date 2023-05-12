# Spam-Detetection-Using-Fedrate-Learning

# Federated Learning with BERT

This repository contains the code for implementing federated learning using BERT (Bidirectional Encoder Representations from Transformers) for text classification. The project demonstrates how to train a BERT-based model on distributed clients while aggregating the model updates on a central server.

## Files

The repository consists of the following files:

1. `server.py`: This file contains the server code for coordinating the federated learning process. It defines a custom aggregation strategy based on the FedAvg algorithm. The server clusters the clients based on their hamming percentage and selects the top models from each cluster to aggregate their weights.

2. `client.py`: This file represents the client code that runs on individual devices participating in the federated learning process. It uses the Flower library to communicate with the server and exchange model parameters. The client code preprocesses the data, initializes the BERT model, and trains the model on its local dataset. It then sends the model parameters back to the server for aggregation.

3. `script.py`: This script provides a convenient way to start the server and multiple clients for federated learning. It sets the required parameters such as the server port, the number of clients, and the file name to run on the clients. It spawns subprocesses to run the server and the specified number of clients simultaneously.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/federated-learning-with-bert.git
```
2. Install the required dependencies:

 ```bash
 pip install -r requirements.txt
 ```
 
3. Prepare the data:

   - Place the dataset file `enron_spam_data_new.csv` in the project directory.

4. Start the federated learning process:

   - Open a terminal and run the script.py:
