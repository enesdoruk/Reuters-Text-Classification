import matplotlib.pyplot as plt

def plot_histogram(cat_doc, type=""):
    lengths = [len(docs) for docs in cat_doc.values()]
    plt.bar(cat_doc.keys(), lengths, color='skyblue')
    plt.xlabel('Categories')
    plt.ylabel('Number of Documents')
    plt.title(f'Number of Documents in Each Category on {type}')
    plt.xticks(rotation=45)
    plt.show()