from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from sklearn.decomposition import PCA
from umap import UMAP
import pandas as pd
import numpy as np
import logging
import os
import json

# Initialize Flask application with configuration
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
socketio = SocketIO(app)

# Configure logging for application
logging.basicConfig(level=logging.INFO)

# Route to render the main HTML page
@app.route('/')
def index():
    """Render the main page with the HTML template."""
    return render_template('index.html')

# Helper function to load pre-generated demo JSON data
def load_demo_json(demo_file):
    """Load pre-generated demo JSON data from the responseData folder."""
    try:
        # Path to the folder containing the JSON files
        response_data_folder = 'demo_file_responseData'
        
        # Build the path to the JSON file based on the demo file name
        json_file_path = os.path.join(response_data_folder, f"{demo_file}.json")
        
        # Check if the file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Demo JSON file for {demo_file} not found.")
        
        # Load and return the JSON data
        with open(json_file_path, 'r') as f:
            return json.load(f)
    
    except Exception as e:
        logging.error(f"Error loading demo file: {e}")
        raise

# Route to upload data and process PCA and UMAP
@app.route('/upload_data', methods=['POST'])
def upload_data():
    try:
        # Get the uploaded files or demo file
        main_file = request.files.get('main_file')
        edge_file = request.files.get('edge_file')
        demo_file = request.form.get('builtin_file')

        # If neither a file nor demo dataset is selected, return an error
        if not main_file and not demo_file:
            logging.error("No file or demo file provided.")
            return jsonify({"error": "Main file or demo file is required"}), 400

        # Handle demo file or uploaded files
        if demo_file:
            # Load demo data from pre-generated JSON
            demo_data = load_demo_json(demo_file)
            pca_data = demo_data['pcaData']
            umap_data = demo_data['umapData']
        else:
            # Handle file upload and process files
            df, labels, edges = handle_file_upload(main_file, edge_file)
            numeric_df = preprocess_data(df)
            pca_data = process_pca(numeric_df, labels, edges)
            umap_data = process_umap(numeric_df, labels, edges)

        # Construct the response data to send back
        response_data = {
            "pcaData": pca_data,
            "umapData": umap_data,
        }

    # #   Save response data to a local file
    #     output_file_path = os.path.join(os.getcwd(), "response_data.json")
    #     with open(output_file_path, 'w') as f:
    #         json.dump(response_data, f, indent=2)

    #     logging.info(f"Response data saved to {output_file_path}")
    #     try:
    #         json.dumps(response_data)  # Test serialization
    #     except TypeError as e:
    #         logging.error(f"Serialization error: {e}")
    #         raise ValueError("Data contains non-serializable types.")

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error processing the file: {e}")
        return jsonify({"error": "Error processing the file"}), 500

# File upload handler
def handle_file_upload(main_file, edge_file):
    """Handle file upload and processing for main and edge files."""
    if not main_file:
        raise ValueError("No main file provided.")
    
    df = pd.read_csv(main_file, header=0)
    labels = df['label'].tolist() if 'label' in df.columns else list(range(1, len(df) + 1))

    edges = []
    # Check if the edge file is provided and process it
    if edge_file and edge_file.filename.strip() != '':
        edge_data = pd.read_csv(edge_file, header=None)
        edges = edge_data.to_numpy().tolist()

    return df, labels, edges

# Refined function to process the main data and edge file
def preprocess_data(df):
    """Standardize column names, optimize data types, and select numeric columns."""
    # Standardize column names
    df = df.rename(columns=lambda x: x.lower())

    # Optimize data types
    df = optimize_dtypes(df)

    # Select only numeric columns
    numeric_columns = [col for col in df.columns if col != 'label']
    numeric_df = df[numeric_columns].select_dtypes(include=[float, int])

    # Convert float columns to float32
    float_columns = numeric_df.select_dtypes(include=['float']).columns
    numeric_df[float_columns] = numeric_df[float_columns].astype("float32")

    # Ensure there are enough numeric columns for algorithm
    if numeric_df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 numeric columns for PCA or UMAP.")
    return numeric_df

def optimize_dtypes(df):
    """Optimize data types in a DataFrame to reduce memory usage."""
    for col in df.columns:
        col_dtype = df[col].dtype

        # Optimize integers
        if col_dtype in ['int64', 'int32', 'int16']:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        # Optimize floats
        elif col_dtype in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Optimize object columns with repeated values to category
        elif col_dtype == 'object' and df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')

    return df

def process_pca(numeric_df, labels, edges):
    # Apply mean-centering for PCA
    numeric_df_centered = numeric_df - numeric_df.mean(axis=0)

    # Perform PCA analysis with the full possible number of components
    max_initial_components = min(numeric_df_centered.shape)
    logging.info(f"Performing PCA with {max_initial_components} components.")
    pca = PCA(n_components=max_initial_components, random_state=42)
    pca_result = pca.fit_transform(numeric_df_centered)

    # Calculate explained variance and cumulative variance
    explained_variance = pca.explained_variance_ratio_
    explained_variance_cumsum = explained_variance.cumsum()

    # Determine the number of components needed to explain at least 95% of variance
    num_components_95_variance = next(
        (i + 1 for i, v in enumerate(explained_variance_cumsum) if v >= 0.95),
        max_initial_components
    )

    # If the total number of components is less than or equal to 100, use all components
    if max_initial_components <= 100:
        num_components = max_initial_components
    else:
        # Otherwise, limit the number of components to either 100 or those explaining 95% variance
        num_components = min(num_components_95_variance, 160)

    # Filter the PCA results and variance data to the selected number of components
    pca_result = pca_result[:, :num_components]
    explained_variance = explained_variance[:num_components]
    explained_variance_cumsum = explained_variance_cumsum[:num_components]

    # Create pairs of principal components and their combined variances
    pairs = [(i + 1, j + 1) for i in range(num_components) for j in range(i + 1, num_components)]
    sorted_pairs_with_variance = sorted(
        [(pair, explained_variance[pair[0] - 1] + explained_variance[pair[1] - 1])
         for pair in pairs],
        key=lambda x: -x[1]
    )
    sorted_pairs = [{"pair": pair, "combined_variance": variance * 100} for pair, variance in sorted_pairs_with_variance]

    # Prepare PCA data for the response
    pca_data = {f'pca{i+1}': pca_result[:, i].tolist() for i in range(num_components)}
    pca_data['num_components'] = num_components
    pca_data['sorted_pairs'] = sorted_pairs
    pca_data['explained_variance_percentage'] = [v * 100 for v in explained_variance]
    pca_data['explained_variance_cumulative'] = explained_variance_cumsum[-1] * 100
    pca_data['labels'] = labels  # Include labels
    pca_data['edges'] = edges    # Include edges
    pca_data['color_map'] = process_pca_color_map(pca_result)

    return pca_data

def process_umap(numeric_df, labels, edges):
    # Adjust n_neighbors based on dataset size
    n_neighbors = min(8, numeric_df.shape[0] - 1)
    n_components = min(5, numeric_df.shape[1])  # Target number of dimensions
    subsample_size = min(5000, numeric_df.shape[0])
    if subsample_size < 2:
        raise ValueError("The dataset is too small to process UMAP.")

    # Randomly sample a subset of the data
    subsample = numeric_df.sample(n=subsample_size, random_state=42)  # Subsample 5000 rows

    # Extract the indices of the subsample
    subsample_indices = subsample.index

    # Subsample the labels based on the subsample indices
    subsample_labels = [labels[i] for i in subsample_indices]

    # Filter edges based on the subsample
    subsample_edges = [
        [subsample_indices.get_loc(edge[0]), subsample_indices.get_loc(edge[1])]
        for edge in edges if edge[0] in subsample_indices and edge[1] in subsample_indices
    ]

    # UMAP embedding
    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=42,
        min_dist=0.5,
        metric="cosine",
        n_jobs=-1,  # Enable parallel computation
        low_memory=True # Reduce memory usage and speed up computation
    )
    umap_result = umap.fit_transform(subsample)

    # Calculate projection densities for sorting
    densities = [numeric_df.iloc[:, i].var() for i in range(n_components)]
    sorted_pairs_with_density = sorted(
        [{"pair": (i + 1, j + 1), "combined_density": densities[i] + densities[j]}
         for i in range(n_components) for j in range(i + 1, n_components)],
        key=lambda x: -x["combined_density"]
    )

    # Prepare UMAP data
    umap_data = {f'umap{i+1}': umap_result[:, i].tolist() for i in range(n_components)}
    umap_data['num_components'] = n_components
    umap_data['sorted_pairs'] = sorted_pairs_with_density
    umap_data['projection_densities'] = [d for d in densities]
    umap_data['color_map'] = process_umap_colormap(densities, umap_result)
    umap_data['labels'] = subsample_labels  # Assign the subsample labels here
    umap_data['edges'] = subsample_edges  # Assign the filtered edges here

    # Return the UMAP data along with the subsample labels and edges
    return umap_data

def process_pca_color_map(pca_result):
    # Generate color map based on the pc1
    pc1_values = np.array(pca_result[:, 0])
    min_pc1, max_pc1 = pc1_values.min(), pc1_values.max()

    # Normalize
    color_map = [float((value - min_pc1)) / float((max_pc1 - min_pc1)) for value in pc1_values]
    return color_map

def process_umap_colormap(densities, umap_result):
    # Generate color map based on highest density element
    highest_density_idx = np.argmax(densities)
    highest_density_component = umap_result[:, highest_density_idx]

    # Normalize
    color_map = (highest_density_component - highest_density_component.min()) / (
        highest_density_component.max() - highest_density_component.min())

    return color_map.tolist()

# Start the Flask application
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=9000, allow_unsafe_werkzeug=True)
