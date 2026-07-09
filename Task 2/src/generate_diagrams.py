import os
import matplotlib.pyplot as plt

def draw_block_diagram(title, blocks, filepath):
    fig, ax = plt.subplots(figsize=(6, len(blocks) * 1.2))
    ax.axis('off')
    
    # Background styling
    fig.patch.set_facecolor('#ffffff')
    
    box_style = dict(boxstyle="round,pad=0.5", fc="#2980b9", ec="#1f618d", lw=2)
    text_style = dict(color="white", ha="center", va="center", fontsize=11, fontweight='bold')
    
    y = len(blocks)
    
    for i, block in enumerate(blocks):
        # Draw block box
        ax.text(0.5, y - 0.5, block, bbox=box_style, **text_style)
        
        # Draw arrow down to the next block
        if i < len(blocks) - 1:
            ax.annotate('', xy=(0.5, y - 1.05), xytext=(0.5, y - 0.95),
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))
        y -= 1.2
        
    plt.title(title, fontsize=14, fontweight='bold', pad=20, color="#2c3e50")
    plt.xlim(0, 1)
    plt.ylim(0, len(blocks) * 1.2)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Diagram saved to {filepath}")

def main():
    # 1. Classical Pipeline Diagram
    classical_blocks = [
        "Raw IMDb Text Reviews\n(25k Pos / 25k Neg)",
        "Text Preprocessing\n(HTML Strip, Punctuation Filter, Stopwords)",
        "Frozen Dataset Split\n(Train / Val / Test)",
        "TF-IDF Vectorization\n(Vocabulary Size = 10,000)",
        "Linear Classifiers\n(Logistic Regression / SVM)"
    ]
    draw_block_diagram("Classical Machine Learning Pipeline", classical_blocks, "../plots/classical_pipeline.png")
    
    # 2. LSTM Architecture Diagram
    lstm_blocks = [
        "Tokenized Input Sequence\n(Max Sequence Length = 150)",
        "Embedding Layer\n(Trainable Scratch vs. Static GloVe 50d)",
        "LSTM Layer\n(64 or 128 units, recurrent dropout=0.0)",
        "Dense ReLu Layer\n(Dropout = 0.3 or 0.5)",
        "Sigmoid Classification Layer\n(Binary Sentiment Prediction)"
    ]
    draw_block_diagram("LSTM Deep Learning Architecture", lstm_blocks, "../plots/lstm_architecture.png")

if __name__ == "__main__":
    main()
