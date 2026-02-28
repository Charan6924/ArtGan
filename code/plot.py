import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('logs.csv')

plt.plot(df['epoch'],df['train_loss'])
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('train_loss')
plt.clf()

plt.plot(df['epoch'],df['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('val_loss')
plt.clf()

plt.plot(df['epoch'],df['style_acc'])
plt.title('Style Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('style_acc')
plt.clf()

plt.plot(df['epoch'],df['artist_acc'])
plt.title('Artist Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('artist_acc')
plt.clf()

