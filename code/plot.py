import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs.csv')

print(df.head())

lr = df['lr']
plt.plot(lr, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Time')
plt.savefig('lr.png')
plt.clf()

plt.plot(df['train_loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train loss')
plt.legend()
plt.savefig('train_loss.png')
plt.clf()

plt.plot(df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Val loss')
plt.legend()
plt.savefig('val_loss.png')
plt.clf()

plt.plot(df['style_acc'], label='Style Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Style Accuracy')
plt.legend()
plt.savefig('style_acc.png')
plt.clf()

plt.plot(df['artist_acc'],label = 'Artist Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Artist Accuracy')
plt.legend()
plt.savefig('artist_acc.png')
plt.clf()