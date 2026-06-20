import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.proj  = nn.Linear(in_channels, hidden_channels) if in_channels > hidden_channels else nn.Identity()
        actual_in  = hidden_channels if in_channels > hidden_channels else in_channels
        self.conv1 = SAGEConv(actual_in, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        out  = model(batch.x, batch.edge_index)
        pred = out[:batch.batch_size].argmax(dim=1)
        correct += (pred == batch.y[:batch.batch_size]).sum().item()
        total   += batch.batch_size
    return correct / total


def train(data, x_features, in_channels, num_classes, model_name="GraphSAGE",
          epochs=300, patience=20, lr=1e-3,
          num_neighbors=[15, 10], batch_size=512):

    data.x = torch.tensor(x_features, dtype=torch.float)

    train_loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                  batch_size=batch_size, input_nodes=data.train_mask, shuffle=True)
    val_loader   = NeighborLoader(data, num_neighbors=[-1, -1],
                                  batch_size=batch_size, input_nodes=data.val_mask)
    test_loader  = NeighborLoader(data, num_neighbors=[-1, -1],
                                  batch_size=batch_size, input_nodes=data.test_mask)

    model     = GraphSAGE(in_channels, 256, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    history = {'train': [], 'val': [], 'test': [], 'loss': []}
    best_val_acc, best_test_acc, patience_counter = 0.0, 0.0, 0

    pbar = tqdm(range(1, epochs + 1), desc=model_name)
    for epoch in pbar:
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss  = epoch_loss / n_batches
        train_acc = evaluate(model, train_loader)
        val_acc   = evaluate(model, val_loader)
        test_acc  = evaluate(model, test_loader)

        pbar.set_postfix(loss=f"{avg_loss:.4f}", val=f"{val_acc:.4f}", test=f"{test_acc:.4f}")
        history['train'].append(train_acc)
        history['val'].append(val_acc)
        history['test'].append(test_acc)
        history['loss'].append(avg_loss)

        if val_acc > best_val_acc:
            best_val_acc, best_test_acc = val_acc, test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.close()
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest Val Acc:  {best_val_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'],   label='Val')
    plt.plot(history['test'],  label='Test')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], color='red', label='Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend(); plt.grid(True)

    plt.tight_layout(); plt.show()
    return best_val_acc, best_test_acc
