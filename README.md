# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset


## DESIGN STEPS
### STEP 1: 
Load data, create word/tag mappings, and group sentences.

### STEP 2: 
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3: 
Define dataset and DataLoader for batching.

### STEP 4: 
Build a bidirectional LSTM model for sequence tagging.

### STEP 5: 
Train the model over multiple epochs, tracking loss.

### STEP 6: 
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.





## PROGRAM

### Name:Navinkumar V

### Register Number: 212223230141

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)
        
model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses,val_losses=[],[]
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, val Loss = {val_loss:.4f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
                        pred_tags.append(idx2tag[preds[i][j].item()])

# Run training and evaluation
train_losses, val_losses = train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3)
evaluate_model(model, test_loader, X_test, y_test)

# Plot loss
print('Name:Navinkumar V')
print('Register Number: 212223230141')
history_df = pd.DataFrame({"loss": train_losses, "val_loss": val_losses})
history_df.plot(title="Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Inference and prediction
i = 125
model.eval()
sample = X_test[i].unsqueeze(0).to(device)
output = model(sample)
preds = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
true = y_test[i].numpy()

print('Name:Navinkumar V')
print('Register Number: 212223230141')
print("{:<15} {:<10} {}\n{}".format("Word", "True", "Pred", "-" * 40))
for w_id, true_tag, pred_tag in zip(X_test[i], y_test[i], preds):
    if w_id.item() != word2idx["ENDPAD"]:
        word = words[w_id.item() - 1]
        true_label = tags[true_tag.item()]
        pred_label = tags[pred_tag]
        print(f"{word:<15} {true_label:<10} {pred_label}")

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="513" height="421" alt="image" src="https://github.com/user-attachments/assets/d4e95eb1-4d35-4527-ad06-207dbd52d7b7" />


### Sample Text Prediction

<img width="295" height="353" alt="image" src="https://github.com/user-attachments/assets/ef8cb82d-53b4-4690-bc60-8b4efaac2df0" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
