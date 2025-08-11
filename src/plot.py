import matplotlib.pyplot as plt


models = ['LSTM', 'ResNet-34', 'CNN+Attention']
acc_pd = [81.2, 82.9, 84.7]
acc_pi = [74.5, 75.7, 77.6]

plt.figure(figsize=(8, 5))
plt.bar(models, acc_pd, label='Patient-Dependent', width=0.35, align='center')
plt.bar(models, acc_pi, bottom=acc_pd, label='Patient-Independent', width=0.35, align='edge')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison Across Models')
plt.legend()
plt.tight_layout()
plt.show()


