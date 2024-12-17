from flask import Flask, render_template, request, redirect, url_for, flash
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from xgboost import Booster

def debug_log(title, content):
    with open(os.path.join(MODEL_PATH, 'debug_log.txt'), 'a') as f:
        f.write(f"{title}:\n{content}\n\n")



# Flask app setup
app = Flask(__name__)
app.secret_key = "e9c79212962873f33944fdb76faccbc7"  

# Paths to models and scalers
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

# Load combined model from JSON
combined_model_path = os.path.join(MODEL_PATH, 'combined_model.json')
combined_model = Booster()
combined_model.load_model(combined_model_path)
combined_scaler = joblib.load(os.path.join(MODEL_PATH, 'combined_scaler.joblib'))

# Load top features model from JSON
top_features_model_path = os.path.join(MODEL_PATH, 'top_features_model.json')
top_features_model = Booster()
top_features_model.load_model(top_features_model_path)
top_features_scaler = joblib.load(os.path.join(MODEL_PATH, 'top_features_scaler.joblib'))

def load_spectrum(filepath):
    """Load mass spectrum from CSV file."""
    import pandas as pd
    try:
        spectrum = pd.read_csv(filepath)
        
        # Handle cases where expected column headers are missing
        if 'X(Thompsons)' in spectrum.columns and 'Y(Counts)' in spectrum.columns:
            mz = spectrum['X(Thompsons)'].values
            intensities = spectrum['Y(Counts)'].values
        else:
            # Attempt to identify columns dynamically
            mz = spectrum.iloc[:, 1].values  # Assuming second column is X(Thompsons)
            intensities = spectrum.iloc[:, 2].values  # Assuming third column is Y(Counts)
        
        return mz, intensities
    except Exception as e:
        flash(f"Error loading file {filepath}: {e}", "error")
        return None, None


def align_spectra(mz1, intensities1, mz2, intensities2, n_points=1000):
    """Align two spectra to a common m/z axis using interpolation."""
    from scipy.interpolate import interp1d
    try:
        min_mz = max(np.min(mz1), np.min(mz2))
        max_mz = min(np.max(mz1), np.max(mz2))
        common_mz = np.linspace(min_mz, max_mz, n_points)

        f1 = interp1d(mz1, intensities1, kind='linear', bounds_error=False, fill_value=0)
        f2 = interp1d(mz2, intensities2, kind='linear', bounds_error=False, fill_value=0)

        int1_aligned = f1(common_mz)
        int2_aligned = f2(common_mz)

        return common_mz, int1_aligned, int2_aligned
    except Exception as e:
        flash(f"Error aligning spectra: {e}", "error")
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page for model selection and file upload."""
    if request.method == 'POST':
        # Get form data
        selected_model = request.form.get('model')
        spectrum_file1 = request.files.get('spectrum1')
        spectrum_file2 = request.files.get('spectrum2')

        # Validate input
        if not selected_model or not spectrum_file1 or not spectrum_file2:
            flash("Please select a model and upload two spectrum files.", "error")
            return redirect(url_for('home'))

        # Save uploaded files
        filepath1 = os.path.join("static/uploads", spectrum_file1.filename)
        filepath2 = os.path.join("static/uploads", spectrum_file2.filename)
        spectrum_file1.save(filepath1)
        spectrum_file2.save(filepath2)

        # Redirect to results page
        return redirect(url_for('results', model=selected_model, filepath1=filepath1, filepath2=filepath2))

    return render_template('home.html')

@app.route('/results')
def results():
    """Results page for displaying predictions and spectra plots."""
    model_name = request.args.get('model')
    filepath1 = request.args.get('filepath1')
    filepath2 = request.args.get('filepath2')

    # Load spectrum data
    mz1, intensities1 = load_spectrum(filepath1)
    mz2, intensities2 = load_spectrum(filepath2)
    if mz1 is None or mz2 is None:
        return redirect(url_for('home'))

    # Align spectra
    common_mz, int1_aligned, int2_aligned = align_spectra(mz1, intensities1, mz2, intensities2)

    # Preprocess data and make predictions
    if model_name == 'cosine':
        model = combined_model
        scaler = combined_scaler
        features = np.array([[np.dot(int1_aligned, int2_aligned) /
                              (np.linalg.norm(int1_aligned) * np.linalg.norm(int2_aligned))]])
    elif model_name == 'top_features':
        model = top_features_model
        scaler = top_features_scaler
        area1 = np.trapz(int1_aligned, common_mz)
        area2 = np.trapz(int2_aligned, common_mz)
        features = np.array([[
            np.dot(int1_aligned, int2_aligned) / (np.linalg.norm(int1_aligned) * np.linalg.norm(int2_aligned)),
            np.corrcoef(int1_aligned, int2_aligned)[0, 1],
            min(area1 / area2, area2 / area1) if area1 > 0 and area2 > 0 else 0
        ]])
    else:
        flash("Invalid model selection.", "error")
        return redirect(url_for('home'))

    # Scale features
    scaled_features = scaler.transform(features)

    # Convert to DMatrix and predict
    from xgboost import DMatrix
    dmatrix_features = DMatrix(scaled_features)
    prediction = model.predict(dmatrix_features)

    # Generate plot
    plot_url = plot_comparison(common_mz, int1_aligned, int2_aligned)

    return render_template('results.html', model_name=model_name, plot_url=plot_url, prediction=prediction)

def plot_comparison(mz, int1, int2):
    """Plot and return the aligned spectra as a base64 string."""
    plt.figure(figsize=(10, 6))
    plt.plot(mz, int1, label="Spectrum 1 (Aligned)", alpha=0.7)
    plt.plot(mz, int2, label="Spectrum 2 (Aligned)", alpha=0.7)
    plt.xlabel("m/z (Thompson)")
    plt.ylabel("Intensity (Normalized)")
    plt.title("Aligned Spectra Comparison")
    plt.legend()
    plt.grid(True)

    # Save plot to a string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close()

    return f"data:image/png;base64,{plot_data}"

if __name__ == '__main__':
    # Create static/uploads directory if not exists
    os.makedirs("static/uploads", exist_ok=True)

    # Run the app
    app.run(debug=True)
