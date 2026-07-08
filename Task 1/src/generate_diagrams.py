import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture(layers, title, save_path):
    """
    Draws a vertical block diagram representing the neural network architecture.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) * 1.5 + 1)
    
    # Hide axes
    ax.axis('off')
    
    # Title
    ax.text(5, len(layers) * 1.5 + 0.5, title, fontsize=14, fontweight='bold', ha='center')
    
    y = len(layers) * 1.5 - 0.5
    for i, layer in enumerate(layers):
        l_type = layer.get('type', 'Layer')
        name = layer.get('name', '')
        shape = layer.get('shape', '')
        details = layer.get('details', '')
        
        # Select color based on layer type
        if l_type == 'input':
            color = '#a2d2ff' # Light Blue
        elif l_type == 'conv':
            color = '#bde0fe' # Soft Blue
        elif l_type == 'pool':
            color = '#ffc8dd' # Soft Pink
        elif l_type == 'dense':
            color = '#ffafcc' # Pink
        elif l_type == 'norm':
            color = '#e2e2e2' # Light Gray
        elif l_type == 'dropout':
            color = '#fde2e4' # Soft Red
        elif l_type == 'aug':
            color = '#d8f3dc' # Soft Green
        else:
            color = '#ffffff'
            
        # Draw layer box
        box = patches.FancyBboxPatch(
            (1.5, y - 0.4), 7.0, 0.8,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor="black", facecolor=color
        )
        ax.add_patch(box)
        
        # Text inside box
        ax.text(5, y + 0.15, name, fontsize=10, fontweight='bold', ha='center')
        ax.text(5, y - 0.2, f"Shape: {shape} | {details}", fontsize=8.5, ha='center', style='italic')
        
        # Draw arrow to next layer
        if i < len(layers) - 1:
            ax.annotate('', xy=(5, y - 0.7), xytext=(5, y - 0.4),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            
        y -= 1.5
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to {save_path}")

def main():
    # Baseline CNN Layers
    baseline_layers = [
        {'type': 'input', 'name': 'Input Layer', 'shape': '32 x 32 x 3', 'details': 'Normalized RGB Image'},
        {'type': 'conv', 'name': 'Conv2D (32 filters)', 'shape': '32 x 32 x 32', 'details': 'Kernel: 3x3 | Activation: ReLU'},
        {'type': 'pool', 'name': 'MaxPooling2D', 'shape': '16 x 16 x 32', 'details': 'Pool: 2x2 | Stride: 2'},
        {'type': 'conv', 'name': 'Conv2D (64 filters)', 'shape': '16 x 16 x 64', 'details': 'Kernel: 3x3 | Activation: ReLU'},
        {'type': 'pool', 'name': 'MaxPooling2D', 'shape': '8 x 8 x 64', 'details': 'Pool: 2x2 | Stride: 2'},
        {'type': 'conv', 'name': 'Conv2D (128 filters)', 'shape': '8 x 8 x 128', 'details': 'Kernel: 3x3 | Activation: ReLU'},
        {'type': 'pool', 'name': 'MaxPooling2D', 'shape': '4 x 4 x 128', 'details': 'Pool: 2x2 | Stride: 2'},
        {'type': 'dense', 'name': 'Flatten & Dense (128 units)', 'shape': '128', 'details': 'Activation: ReLU'},
        {'type': 'dense', 'name': 'Dense (10 units)', 'shape': '10', 'details': 'Activation: Softmax (Output)'}
    ]
    
    # Final Customized CNN Layers (combining winning techniques)
    final_layers = [
        {'type': 'input', 'name': 'Input Layer', 'shape': '32 x 32 x 3', 'details': 'Normalized RGB Image'},
        {'type': 'aug', 'name': 'Data Augmentation Block', 'shape': '32 x 32 x 3', 'details': 'Random Flip | Random Rotation 0.1 | Random Translation 0.1'},
        
        # Block 1
        {'type': 'conv', 'name': 'Conv2D (32 filters)', 'shape': '32 x 32 x 32', 'details': 'Kernel: 3x3 | L2 Weight Decay 1e-4'},
        {'type': 'norm', 'name': 'Batch Normalization', 'shape': '32 x 32 x 32', 'details': 'Standardizes features across batch'},
        {'type': 'pool', 'name': 'MaxPooling2D & Dropout', 'shape': '16 x 16 x 32', 'details': 'Pool: 2x2 | Dropout: 0.3'},
        
        # Block 2
        {'type': 'conv', 'name': 'Conv2D (64 filters)', 'shape': '16 x 16 x 64', 'details': 'Kernel: 3x3 | L2 Weight Decay 1e-4'},
        {'type': 'norm', 'name': 'Batch Normalization', 'shape': '16 x 16 x 64', 'details': 'Standardizes features across batch'},
        {'type': 'pool', 'name': 'MaxPooling2D & Dropout', 'shape': '8 x 8 x 64', 'details': 'Pool: 2x2 | Dropout: 0.3'},
        
        # Block 3
        {'type': 'conv', 'name': 'Conv2D (128 filters)', 'shape': '8 x 8 x 128', 'details': 'Kernel: 3x3 | L2 Weight Decay 1e-4'},
        {'type': 'norm', 'name': 'Batch Normalization', 'shape': '8 x 8 x 128', 'details': 'Standardizes features across batch'},
        {'type': 'pool', 'name': 'MaxPooling2D & Dropout', 'shape': '4 x 4 x 128', 'details': 'Pool: 2x2 | Dropout: 0.3'},
        
        # Flatten & FC Block
        {'type': 'dense', 'name': 'Flatten & Dense (128 units)', 'shape': '128', 'details': 'L2 Weight Decay 1e-4'},
        {'type': 'norm', 'name': 'Batch Normalization', 'shape': '128', 'details': 'Standardizes FC activations'},
        {'type': 'dropout', 'name': 'Dropout (0.5)', 'shape': '128', 'details': 'Prevents dense layer co-adaptation'},
        
        # Output
        {'type': 'dense', 'name': 'Dense (10 units)', 'shape': '10', 'details': 'Activation: Softmax (Output)'}
    ]
    
    draw_architecture(baseline_layers, "Baseline CNN Model Architecture", "plots/baseline_architecture.png")
    draw_architecture(final_layers, "Final Customized CNN Model Architecture", "plots/final_architecture.png")

if __name__ == "__main__":
    main()
