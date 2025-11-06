import matplotlib.pyplot as plt

def plot_training(log_history, save_path=None):
    train_loss = log_history['train_loss']
    val_loss = log_history['val_loss']
    epochs = range(1, len(train_loss)+1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
