import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv(r'C:\Users\Hemant Sri Kumar\Desktop\Prodigy\Task_02\train.csv')
survival_pivot = (
    train_df.groupby(['Pclass', 'Sex'])['Survived'].mean()
    .unstack()
    * 100
).round(1)

plt.figure(figsize=(7, 5))

sns.heatmap(
    survival_pivot,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=.5,
)

plt.title('Survival Rate (%) by Pclass and Sex')
plt.show()