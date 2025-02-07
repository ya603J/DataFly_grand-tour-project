# DataFly: Visualizing High-Dimensional Data

**DataFly** is a web-based tool designed to help users explore high-dimensional datasets. By leveraging dimensionality reduction techniques like PCA and UMAP, it projects complex data into an intuitive 2D space, making it easier to analyze and interpret. The tool allows users to view different dimensional projections of their data, helping to uncover hidden patterns or relationships within the original dataset.

---

## Key Features

- **File Upload**: Upload your datasets (CSV or graph files) for visualization.
- **Demo Datasets**: Explore built-in datasets to see the tool in action.
- **Dimensionality Reduction**: Use PCA or UMAP to reduce dimensions and discover patterns.
- **Interactive Controls**: Adjust projections and parameters dynamically.
- **Smooth Transitions**: Experience seamless updates as projections change.
- **Graph Support**: Visualize graph data by connecting points with optional edges.
- **Label Management**: Toggle labels on the visualization to focus on specific data attributes. (Delayed for now)

---

## Upload File Requirements

| File       | Headers Required | Mandatory Columns                          | File Format            | Optional or Required |
|------------|------------------|--------------------------------------------|------------------------|-----------------------|
| `main_file` | Yes              | At least 2 numeric columns; optional label column | Valid CSV with headers | Required              |
| `edge_file` | No               | Each row as `node1,node2`                  | Valid CSV without headers | Optional              |

---
## Visit the website
https://grand-tour-project.onrender.com/

## How to Use

1. **Upload Data**:
   - Upload a CSV or node file as your main data source (required).
   - Optionally, upload an edge file for graph-based visualizations.
   - Alternatively, select a demo dataset from the dropdown menu.

2. **Select an Algorithm**:
   - Choose **PCA** or **UMAP** for dimensionality reduction.

3. **Adjust Projections**:
   - Use sliders to select projection dimensions (e.g., Projection 1 vs Projection 2).

4. **Visualize**:
   - Click **Go** to render the 2D visualization.
   - Use **DataFly** to explore additional features.
   - Toggle labels using the **Toggle Labels** button. (Delayed for now)

5. **DataFly**:
   - To pause and view the current pair, hover over the **DataFly** button. The visualization will pause at the current pair. Removing the hover will resume the transition to the next pairs.
   - The **DataFly** visualization automatically pauses when you:
     - Adjust the sliders to select specific projections.
     - Click the **Go** button to view the selected pair.
     - Switch the algorithm.

6. **Reset Data**:
   - Click **Reset** to clear visualizations and start fresh.

---

## Project Structure

- **`app.py`**: Backend logic for processing data, performing dimensionality reduction, and serving results.
- **`templates/index.html`**: Frontend HTML template for user interactions.
- **`static/style.css`**: Styling for the user interface.
- **`static/script.js`**: Handles interactivity and visualization logic.

---

## Demo Datasets

- **word2vec_100D_10k**: Word2vec dataset with 100 dimensions and labels, ideal demonstration of program validity.
- **dw256A_10D_1024**: Large graph dataset with both nods and edges, ensuring robustness.
- **cca_5D_20k**: Large-scale dataset for testing visualization limits.
- **vocabulary_3D_1k**: Small dataset for a easy and quick testing.

---

## Tips and Best Practices

- To visualize graph data, upload both a node file (with `node_id` and coordinates) and an edge file (with connections).
- If labels are missing in the node file, the tool generates sequential labels automatically.
- Experiment with different projection combinations to uncover insights.

---

## Future Enhancements

- Add support for more algorithms like t-SNE or ISOMAP.
- Enable interactive 3D visualizations and integrate cloud-based storage for seamless handling of larger datasets.
