import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

boxplot_test = pd.read_csv("boxplot_test.csv")
sns.boxplot(x=None, y="number", data = boxplot_test)
plt.show()