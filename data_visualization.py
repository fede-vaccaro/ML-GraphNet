from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def reduct_and_visualize(data_subset, y):
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=600)
    tsne_results = tsne.fit_transform(data_subset)

    df = pd.DataFrame()

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['y'] = y

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", y.max() + 1),
        data=df,
        legend="full",
        alpha=1.0
    )

    plt.show()

