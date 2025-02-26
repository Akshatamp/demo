from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, IsolationForest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import io
import base64
import matplotlib.pyplot as plt
from flask_cors import CORS
import seaborn as sns
import os
import matplotlib
import google.generativeai as genai
import traceback
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, jsonify

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
df = None
X_reg_train_scaled = None
X_reg_test_scaled = None
y_reg_train = None
y_reg_test = None
X_cls_train_scaled = None
X_cls_test_scaled = None
y_cls_train = None
y_cls_test = None
label_encoders = {}
numerical_cols = []
categorical_cols = []
scaler = None  # Scaler to be used for scaling data

# Configure the Generative AI model
GOOGLE_API_KEY = "AIzaSyBNEeiQQjvAiGjv81HmPt9kkmHH8RUkfFY" # Replace with your actual key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')


def process_data(file):
    """Processes the uploaded data and prepares it for modeling."""
    global df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test, X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test, label_encoders, numerical_cols, categorical_cols, scaler

    df = pd.read_csv(file)

    # Check for NaN values *before* any processing
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')

    # Drop columns with too many missing values
    df = df.dropna(thresh=len(df) * 0.6, axis=1)

    # Fill missing values
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Select features and target for regression (use last original numerical column)
    original_numerical_cols = [col for col in numerical_cols if col not in label_encoders]
    if len(original_numerical_cols) >= 1:
        X_reg = df[original_numerical_cols[:-1]]  # Use original numerical columns except last
        y_reg = df[original_numerical_cols[-1]]   # Last original numerical column as target
    else:
        X_reg, y_reg = None, None

    # Select features and target for classification (use last encoded categorical column)
    original_categorical_cols = list(label_encoders.keys())
    if original_categorical_cols:
        cls_target = original_categorical_cols[-1]
        X_cls = df.drop(columns=[cls_target])
        y_cls = df[cls_target]
    else:
        X_cls, y_cls = None, None

    # Train-test splits
    if X_reg is not None:
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    if X_cls is not None:
        X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    if X_reg is not None:
        X_reg_train_scaled = scaler.fit_transform(X_reg_train)
        X_reg_test_scaled = scaler.transform(X_reg_test)
    if X_cls is not None:
        X_cls_train_scaled = scaler.fit_transform(X_cls_train)
        X_cls_test_scaled = scaler.transform(X_cls_test)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and data processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        process_data(io.StringIO(file.stream.read().decode("UTF8"), newline=None))  # Passes the file to process_data function
        return jsonify({'message': 'File uploaded and processed successfully'}), 200
    except ValueError as e:
         return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-plot', methods=['POST'])
def generate_plot():
    """Generates 2D and 3D plots without using row numbers as the X-axis."""
    global df

    if df is None:
        return jsonify({"error": "No dataset uploaded. Please upload a file first."}), 400

    data = request.get_json()
    plot_type = data.get("plotType", "").strip()

    try:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Select numeric columns, excluding row numbers
        df_numeric = df.select_dtypes(include=['number'])

        # Ignore row number/index-like columns
        ignore_columns = ["id", "index", "Unnamed: 0"]
        df_numeric = df_numeric.loc[:, ~df_numeric.columns.str.lower().isin(ignore_columns)]

        # Ensure at least two valid columns exist
        if df_numeric.shape[1] < 2:
            return jsonify({"error": "Dataset must contain at least two meaningful numeric columns."}), 400

        # Select the first two numeric columns that are not row numbers
        x_column, y_column = df_numeric.columns[:2]
        df_selected = df[[x_column, y_column]].dropna()

        # Define valid plot types
        valid_plots = [
            "Line Plot", "Scatter Plot", "Bar Chart", "Histogram", 
            "Box Plot", "Heatmap", "Pie Chart", "Contour Plot",
            "3D Scatter Plot", "3D Surface Plot", "3D Wireframe Plot",
            "3D Bar Chart", "3D Contour Plot"
        ]

        if plot_type not in valid_plots:
            return jsonify({"error": f"Invalid plot type '{plot_type}'. Valid options: {', '.join(valid_plots)}"}), 400

        # 2D Plots
        if plot_type == "Line Plot":
            ax.plot(df_selected[x_column], df_selected[y_column], marker='o')
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title("Line Plot")

        elif plot_type == "Scatter Plot":
            ax.scatter(df_selected[x_column], df_selected[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title("Scatter Plot")

        elif plot_type == "Bar Chart":
            ax.bar(df_selected[x_column], df_selected[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title("Bar Chart")

        elif plot_type == "Histogram":
            ax.hist(df_selected[y_column], bins=10)
            ax.set_xlabel(y_column)
            ax.set_title("Histogram")

        elif plot_type == "Box Plot":
            df_selected.boxplot(column=[y_column], ax=ax)
            ax.set_title("Box Plot")

        elif plot_type == "Pie Chart":
            df[y_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title("Pie Chart")

        elif plot_type == "Heatmap":
            sns.heatmap(df_selected.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Heatmap")

        elif plot_type == "Contour Plot":
            X, Y = np.meshgrid(df_selected[x_column], df_selected[y_column])
            Z = np.sin(X) * np.cos(Y)
            ax.contourf(X, Y, Z, cmap='coolwarm')
            ax.set_title("Contour Plot")

        # 3D Plots (Require a third column)
        elif plot_type in ["3D Scatter Plot", "3D Surface Plot", "3D Wireframe Plot", "3D Bar Chart", "3D Contour Plot"]:
            if df_numeric.shape[1] < 3:
                return jsonify({"error": "Dataset must contain at least three numeric columns for 3D plots."}), 400

            z_column = df_numeric.columns[2]
            df_selected = df[[x_column, y_column, z_column]].dropna()
            ax = fig.add_subplot(111, projection='3d')

            X, Y, Z = df_selected[x_column], df_selected[y_column], df_selected[z_column]

            if plot_type == "3D Scatter Plot":
                ax.scatter(X, Y, Z)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_zlabel(z_column)
                ax.set_title("3D Scatter Plot")

            elif plot_type == "3D Surface Plot":
                X_mesh, Y_mesh = np.meshgrid(np.linspace(X.min(), X.max(), 30),
                                             np.linspace(Y.min(), Y.max(), 30))
                Z_mesh = np.sin(X_mesh) * np.cos(Y_mesh)
                ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_zlabel("Generated Z")
                ax.set_title("3D Surface Plot")

            elif plot_type == "3D Wireframe Plot":
                X_mesh, Y_mesh = np.meshgrid(np.linspace(X.min(), X.max(), 30),
                                             np.linspace(Y.min(), Y.max(), 30))
                Z_mesh = np.sin(X_mesh) * np.cos(Y_mesh)
                ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color='black')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_zlabel("Generated Z")
                ax.set_title("3D Wireframe Plot")

            elif plot_type == "3D Bar Chart":
                ax.bar3d(X, Y, np.zeros(len(df)), 1, 1, Z)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_zlabel(z_column)
                ax.set_title("3D Bar Chart")

            elif plot_type == "3D Contour Plot":
                X_mesh, Y_mesh = np.meshgrid(np.linspace(X.min(), X.max(), 30),
                                             np.linspace(Y.min(), Y.max(), 30))
                Z_mesh = np.sin(X_mesh) * np.cos(Y_mesh)
                ax.contour3D(X_mesh, Y_mesh, Z_mesh, 50, cmap='viridis')
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                ax.set_zlabel("Generated Z")
                ax.set_title("3D Contour Plot")

        # Convert plot to base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight')
        plt.close(fig)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

        return jsonify({"plotImage": img_base64, "explanation": f"Plot generated using {x_column} and {y_column}."}), 200

    except Exception as e:
        return jsonify({"error": f"Plot generation failed: {str(e)}"}), 400

def generate_explanation(model_name, results, model_type, data_summary=None):
    """Generates an explanation of the model results using the Gemini Pro."""

    prompt = f"You are an expert data scientist.  Explain the model and results of a {model_type} model called {model_name} on a dataset.  "

    if data_summary:
        prompt += f"Here's a summary of the dataset used:\n{data_summary}\n\n"

    prompt += f"The {model_type} model produced the following results:\n{results}\n\n"
    prompt += "Provide a detailed explanation of these results. Explain what these results mean in the context of the dataset and the chosen algorithm.  Highlight important findings and potential implications."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "An error occurred while generating the explanation."


def execute_regression_model(model, model_name):
    """Executes a regression model and returns results."""
    global X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test

    if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
        raise ValueError('Data not processed. Upload a file first.')

    try:
        model.fit(X_reg_train_scaled, y_reg_train)
        y_pred = model.predict(X_reg_test_scaled)

        results = {
            "MAE": mean_absolute_error(y_reg_test, y_pred),
            "MSE": mean_squared_error(y_reg_test, y_pred),
            "R2 Score": r2_score(y_reg_test, y_pred)
        }

        return results

    except Exception as e:
        raise RuntimeError(f"Error running {model_name}: {str(e)}")


def execute_classification_model(model, model_name):
    """Executes a classification model and returns results and confusion matrix."""
    global X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test, categorical_cols, label_encoders

    if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
        raise ValueError('Data not processed. Upload a file first.')

    try:
        model.fit(X_cls_train_scaled, y_cls_train)
        y_pred = model.predict(X_cls_test_scaled)
        y_pred_proba = model.predict_proba(X_cls_test_scaled)  # Get probabilities for ROC AUC

        metrics = {
            "Accuracy": accuracy_score(y_cls_test, y_pred),
            "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
        }

        # ROC AUC (only if binary classification)
        if len(np.unique(y_cls_train)) == 2:
            try:
                metrics["ROC AUC"] = roc_auc_score(y_cls_test, y_pred_proba[:, 1])
            except Exception as e:
                print(f"Error calculating ROC AUC: {e}")
                metrics["ROC AUC"] = "N/A"  # Or some other placeholder

        # Generate and encode confusion matrix plot
        cm = confusion_matrix(y_cls_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None,
                    yticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()  # Close the plot to free memory

        return metrics, image_base64

    except Exception as e:
        raise RuntimeError(f"Error running {model_name}: {str(e)}")


@app.route('/regression', methods=['POST'])
def run_regression():
    """Runs the selected or custom regression model."""
    model_source = request.json.get('modelSource', 'predefined')  # 'predefined' or 'custom'
    model_name = request.json.get('model', 'Linear Regression')
    custom_code = request.json.get('customCode', '')

    try:
        if model_source == 'predefined':
            models_reg = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "KNN": KNeighborsRegressor(n_neighbors=5),
                "SVM": SVR(),
                "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42)
            }

            if model_name not in models_reg:
                return jsonify({'error': 'Invalid regression model'}), 400

            model = models_reg[model_name]
            results = execute_regression_model(model, model_name)

        elif model_source == 'custom':
            # Execute custom code
            try:
                # Local namespace for executing the custom code safely
                local_namespace = {}

                # Define a custom function within the custom code to create the model
                custom_code_with_model_creation = f"""
def create_custom_model():
    {custom_code}
    return model  # Assuming 'model' is the variable assigned to the model in your custom code

model = create_custom_model()
"""

                # Execute the code within the local namespace
                exec(custom_code_with_model_creation, local_namespace)

                # Retrieve the model created from the local namespace
                model = local_namespace['model']
                model_name = "Custom Model"  # Or try to infer from the custom code

                # Assuming the custom code creates a scikit-learn compatible model
                results = execute_regression_model(model, model_name)

            except Exception as e:
                trace = traceback.format_exc()
                return jsonify({'error': f'Error executing custom code: {str(e)}', 'trace': trace}), 400

        else:
            return jsonify({'error': 'Invalid model source.'}), 400

        # Data Summary for explanation
        data_summary = f"Dataset contains {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns. "
        if numerical_cols:
            data_summary += f"Numerical columns include: {', '.join(numerical_cols)}. "
        if categorical_cols:
            data_summary += f"Categorical columns include: {', '.join(categorical_cols)}."

        explanation = generate_explanation(model_name, results, "regression", data_summary)

        return jsonify({'results': results, 'explanation': explanation}), 200

    except Exception as e:
        trace = traceback.format_exc()  # Get the full traceback
        return jsonify({'error': str(e), 'trace': trace}), 500



@app.route('/classification', methods=['POST'])
def run_classification():
    """Runs the selected or custom classification model."""
    model_source = request.json.get('modelSource', 'predefined')  # 'predefined' or 'custom'
    model_name = request.json.get('model', 'Logistic Regression')
    custom_code = request.json.get('customCode', '')


    try:
        if model_source == 'predefined':
            models_cls = {
                "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "SVM": SVC(probability=True),  # Enable probability for ROC AUC
                "Gaussian Naive Bayes": GaussianNB(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                "Gaussian Process": GaussianProcessClassifier(kernel=RBF(), random_state=42),
                "MLP Classifier": MLPClassifier(random_state=42) # Neural Network
            }

            if model_name not in models_cls:
                return jsonify({'error': 'Invalid classification model'}), 400

            model = models_cls[model_name]
            metrics, image_base64 = execute_classification_model(model, model_name)

        elif model_source == 'custom':
            # Execute custom code
            try:
                # Local namespace for executing the custom code safely
                local_namespace = {}

                # Define a custom function within the custom code to create the model
                custom_code_with_model_creation = f"""
def create_custom_model():
    {custom_code}
    return model  # Assuming 'model' is the variable assigned to the model in your custom code

model = create_custom_model()
"""

                # Execute the code within the local namespace
                exec(custom_code_with_model_creation, local_namespace)

                # Retrieve the model created from the local namespace
                model = local_namespace['model']
                model_name = "Custom Model"  # Or try to infer from the custom code

                # Assuming the custom code creates a scikit-learn compatible model
                metrics, image_base64 = execute_classification_model(model, model_name)

            except Exception as e:
                trace = traceback.format_exc()
                return jsonify({'error': f'Error executing custom code: {str(e)}', 'trace': trace}), 400


        else:
            return jsonify({'error': 'Invalid model source.'}), 400

        # Data Summary for explanation
        data_summary = f"Dataset contains {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns. "
        if numerical_cols:
            data_summary += f"Numerical columns include: {', '.join(numerical_cols)}. "
        if categorical_cols:
            data_summary += f"Categorical columns include: {', '.join(categorical_cols)}."

        explanation = generate_explanation(model_name, metrics, "classification", data_summary)

        return jsonify({'metrics': metrics, 'confusion_matrix': image_base64, 'explanation': explanation}), 200

    except Exception as e:
        trace = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': trace}), 500

@app.route('/clustering', methods=['GET'])
def run_clustering():
    """Runs the clustering algorithm and returns analysis."""
    global df, numerical_cols

    if df is None:
        return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

    # Initialize box_plot_images outside the try block
    box_plot_images = {}  # Initialize it here

    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Setting n_init explicitly avoids a warning
        df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

        # Analyze cluster characteristics
        cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

        # Generate cluster distribution plot
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Cluster', data=df)
        plt.title('Distribution of Clusters')
        buf_cluster_dist = io.BytesIO()
        plt.savefig(buf_cluster_dist, format='png')
        buf_cluster_dist.seek(0)
        cluster_dist_image_base64 = base64.b64encode(buf_cluster_dist.read()).decode('utf-8')
        plt.close()

        # Generate box plots for each numerical column
        #box_plot_images = {}  # Remove from here and initialize on top
        for col in numerical_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Cluster', y=col, data=df)
            plt.title(f'{col} distribution across clusters')
            buf_box = io.BytesIO()
            plt.savefig(buf_box, format='png')
            buf_box.seek(0)
            box_plot_images[col] = base64.b64encode(buf_box.read()).decode('utf-8')
            plt.close()

        # Most distinctive features (as before)
        def get_most_distinctive_features(cluster_id, top_n=3):
            cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
            overall_mean = df[numerical_cols].mean()
            cluster_mean = cluster_data.mean()
            feature_importance = abs(cluster_mean - overall_mean)
            most_important = feature_importance.nlargest(top_n)
            return most_important.index.tolist()

        distinctive_features = {}
        for i in range(3):  # Assuming 3 clusters
            distinctive_features[i] = get_most_distinctive_features(i)


        return jsonify({
            'cluster_analysis': cluster_analysis,
            'cluster_distribution_plot': cluster_dist_image_base64,
            'box_plot_images': box_plot_images,
            'distinctive_features': distinctive_features,
            'numerical_cols': numerical_cols # Return numerical columns
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_data', methods=['POST'])
def add_data():
    """Adds a new data point, generates scatter plots, and provides an explanation
       of its position relative to existing data using Gemini Pro.
    """
    global df, numerical_cols, scaler

    if df is None:
        return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

    try:
        new_data = request.get_json()

        # Validate that the new data contains all the required columns and the values are numeric
        if not all(col in new_data for col in numerical_cols):
            return jsonify({'error': 'Missing required columns in new data.'}), 400
        try:
            new_data_numeric = {col: float(new_data[col]) for col in numerical_cols}
        except ValueError:
            return jsonify({'error': 'Non-numeric values in the new data.'}), 400

        # Add new data to DataFrame
        new_row = pd.Series(new_data_numeric)  # Use the validated numeric values
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Re-run KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

        # Scale the numerical columns using the scaler
        scaled_numerical_cols = scaler.fit_transform(df[numerical_cols])
        df_scaled = pd.DataFrame(scaled_numerical_cols, columns=numerical_cols)

        # Train Isolation Forest model
        iso_forest = IsolationForest(n_estimators=100, random_state=42, contamination='auto')  # contamination handles outlier proportion
        iso_forest.fit(df_scaled[numerical_cols])  # Fit on scaled data

        # Predict outlier status (-1 is outlier, 1 is inlier)
        df['outlier_score'] = iso_forest.decision_function(df_scaled[numerical_cols])  # decision function to score samples
        df['is_outlier'] = iso_forest.predict(df_scaled[numerical_cols])
        is_outlier = (df['is_outlier'].iloc[-1] == -1)  # Determine if the last element is outlier

        scatter_plot_images = {}
        scatter_plot_descriptions = []

        import itertools

        # Generate scatter plots for all combinations of numerical columns
        for col1, col2 in itertools.combinations(numerical_cols, 2):  # All unique pairs
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=col1, y=col2, data=df[:-1], color='pink', label='Original Data')  # all but the last row
            sns.scatterplot(x=[df[col1].iloc[-1]], y=[df[col2].iloc[-1]], color='green', label='New Data Point', marker='o', s=100)  # the last row
            plt.title(f'{col1} vs {col2} with New Data Point')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.legend()

            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            scatter_plot_images[f'{col1}_vs_{col2}'] = image_base64
            plt.close()

            # Generate a textual description of the scatter plot
            original_data = df[:-1]  # All rows except the last one
            new_point = df.iloc[-1]  # The last row (new data point)

            # Calculate distances between the new point and the original data
            distances = np.sqrt((original_data[col1] - new_point[col1])**2 + (original_data[col2] - new_point[col2])**2)
            mean_distance = distances.mean()
            std_distance = distances.std()

            # Describe the position of the new point
            if mean_distance < std_distance:
                position_description = f"The new data point is close to the cluster of original data points in the {col1} vs {col2} scatter plot."
            else:
                position_description = f"The new data point is far from the cluster of original data points in the {col1} vs {col2} scatter plot."

            scatter_plot_descriptions.append(position_description)

        # Generate explanation using Gemini Pro
        explanation = generate_scatter_plot_explanation(scatter_plot_descriptions, is_outlier)

        return jsonify({'scatter_plot_images': scatter_plot_images, 'explanation': explanation}), 200

    except Exception as e:
        trace = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': trace}), 500


def generate_scatter_plot_explanation(scatter_plot_descriptions, is_outlier):
    """Generates an explanation of the scatter plots and outlier status using Gemini Pro."""
    prompt = "You are an expert data scientist. Analyze the following scatter plot descriptions and provide a detailed explanation of the new data point (green dot) relative to the original data (pink dots).\n\n"

    # Add scatter plot descriptions to the prompt
    for description in scatter_plot_descriptions:
        prompt += f"- {description}\n"

    # Add outlier status to the prompt
    if is_outlier:
        prompt += "\nThe new data point is classified as an outlier based on the Isolation Forest algorithm."
    else:
        prompt += "\nThe new data point is not classified as an outlier based on the Isolation Forest algorithm."

    prompt += "\n\nProvide a detailed explanation of the new data point's position relative to the original data, its potential significance, and whether it aligns with the existing data distribution."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "An error occurred while generating the explanation."

if __name__ == '__main__':
    app.run(debug=True)

