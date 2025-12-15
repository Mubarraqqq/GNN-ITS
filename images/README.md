# Images Directory

This directory is for storing images and visual assets used in the GNN Intelligent Tutoring System.

## How to Add Images

### 1. **Concept Diagrams** (`concepts/`)
Add PNG/SVG images of GNN concepts:
- Graph structures
- Neural network architectures
- Algorithm visualizations

Example usage in the Learn tab:
```python
if os.path.exists("images/concepts/graph_structure.png"):
    st.image("images/concepts/graph_structure.png", caption="Graph Structure Example")
```

### 2. **Progress Charts** (`progress/`)
Custom visualizations and infographics for:
- Learning progress
- Performance breakdowns
- Mastery indicators

### 3. **Banners** (`banners/`)
Header/banner images for:
- Tab sections
- Assessment covers
- Objective introductions

### 4. **Icons** (`icons/`)
Small icon images for:
- Concept badges
- Difficulty levels
- Achievement badges

## Recommended Image Formats

- **PNG**: Best for diagrams, icons, and transparent backgrounds
- **SVG**: Vector graphics that scale perfectly
- **JPG**: For photographs and complex images

## Recommended Image Sizes

- Banners: 1200 x 300 pixels
- Concept diagrams: 600 x 400 pixels
- Icons: 64 x 64 to 256 x 256 pixels
- Progress visualizations: 800 x 400 pixels

## Integration Tips

1. **Store images in organized subdirectories** for easy management
2. **Use descriptive filenames** (e.g., `gnn_architecture_layer1.png`)
3. **Add alt text** for accessibility
4. **Compress images** to keep app responsive
5. **Use relative paths** when loading images

## Example Code for Adding Images

```python
import os
from PIL import Image

# Load and display an image
if os.path.exists("images/concepts/gnn_basics.png"):
    img = Image.open("images/concepts/gnn_basics.png")
    st.image(img, caption="Graph Neural Network Basics", use_column_width=True)

# Add multiple images in columns
col1, col2 = st.columns(2)
with col1:
    st.image("images/concepts/concept1.png", caption="Concept 1")
with col2:
    st.image("images/concepts/concept2.png", caption="Concept 2")
```

## Free Resources for Images

- **Diagrams**: [Lucidchart](https://www.lucidchart.com), [Draw.io](https://draw.io)
- **Icons**: [Flaticon](https://www.flaticon.com), [Noun Project](https://thenounproject.com)
- **Stock Photos**: [Unsplash](https://unsplash.com), [Pexels](https://pexels.com)
- **Charts**: Plotly (already integrated), [Matplotlib](https://matplotlib.org)

## Adding Images to Your App

To integrate images in the app, modify `app.py`:

```python
import os
from PIL import Image

def display_concept_image(concept_name: str):
    """Display a concept image if it exists"""
    image_path = f"images/concepts/{concept_name}.png"
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True, caption=f"{concept_name} Illustration")
```

Then use it in tabs:
```python
with tab_learn:
    display_concept_image("graph_neural_networks")
```
