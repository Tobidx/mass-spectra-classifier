from flask import Blueprint, render_template, request, jsonify, current_app
import os
import logging
from app.utils.preprocessing import load_spectrum, align_spectra, calculate_cosine_similarity, calculate_top_features
from app.utils.visualization import create_comparison_plot

# Initialize blueprint and logger
main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@main.route('/')
def index():
    """Home page for model selection and file upload"""
    try:
        available_models = current_app.model_manager.get_available_models()
        logger.info(f"Available models for index: {available_models}")
        return render_template('index.html', models=available_models)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('index.html', models={}, error="Error loading models")

@main.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    logger.debug("Starting prediction request")
    try:
        if 'spectrum1' not in request.files or 'spectrum2' not in request.files:
            logger.error("Missing spectrum files")
            return render_template('index.html',
                                models=current_app.model_manager.get_available_models(),
                                error='Both spectra files are required')

        model_type = request.form.get('model_type')
        available_models = current_app.model_manager.get_available_models()
        
        if not model_type or model_type not in available_models:
            logger.error(f"Invalid model type: {model_type}")
            return render_template('index.html',
                                models=available_models,
                                error='Invalid model selection')

        spectrum1 = request.files['spectrum1']
        spectrum2 = request.files['spectrum2']
        
        # Load spectra
        mz1, int1 = load_spectrum(spectrum1)
        mz2, int2 = load_spectrum(spectrum2)
        
        if mz1 is None or mz2 is None:
            logger.error(f"Failed to load spectra: {spectrum1.filename}, {spectrum2.filename}")
            return render_template('index.html',
                                models=available_models,
                                error='Error loading spectra')

        # Align spectra
        common_mz, int1_aligned, int2_aligned = align_spectra(mz1, int1, mz2, int2)
        if common_mz is None:
            return render_template('index.html',
                                models=available_models,
                                error='Error aligning spectra')

        # Calculate features
        if 'top3' in model_type:
            features = calculate_top_features(common_mz, int1_aligned, int2_aligned)
        else:
            features = calculate_cosine_similarity(common_mz, int1_aligned, int2_aligned)

        if features is None:
            return render_template('index.html',
                                models=available_models,
                                error='Error calculating features')

        # Create plot
        plot_data = create_comparison_plot(
            mz1, int1, mz2, int2,
            spectrum1.filename,
            spectrum2.filename
        )

        # Make prediction
        try:
            prediction, probability = current_app.model_manager.predict(model_type, features)
            logger.debug(f"Prediction: {prediction}, Probability: {probability}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return render_template('index.html',
                                models=available_models,
                                error='Error making prediction')

        # Prepare response data
        response_data = {
            'prediction': prediction,
            'probability': float(probability),
            'features': features,
            'plot_data': plot_data,
            'model_type': model_type,
            'model_name': available_models[model_type],
            'spectrum1_name': spectrum1.filename,
            'spectrum2_name': spectrum2.filename,
            'models': available_models
        }

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(response_data)
            
        return render_template('results.html', **response_data)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        error_message = f"Error processing request: {str(e)}"
        return render_template('index.html',
                            models=current_app.model_manager.get_available_models(),
                            error=error_message)

@main.route('/about')
def about():
    """About page"""
    try:
        return render_template('about.html',
            title='About Mass Spectrometry Classifier',
            description="Mass spectrometry comparison tool using machine learning models",
            model_info={
                'xgboost_cosine': {
                    'name': 'XGBoost with Cosine Similarity',
                    'description': 'Uses XGBoost algorithm with cosine similarity as the single feature.'
                },
                'xgboost_top3': {
                    'name': 'XGBoost with Top 3 Features',
                    'description': 'Uses XGBoost algorithm with three features: cosine similarity, correlation, and area ratio.'
                },
                'rf_cosine': {
                    'name': 'Random Forest with Cosine Similarity',
                    'description': 'Uses Random Forest algorithm with cosine similarity as the single feature.'
                },
                'rf_top3': {
                    'name': 'Random Forest with Top 3 Features',
                    'description': 'Uses Random Forest algorithm with three features: cosine similarity, correlation, and area ratio.'
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in about route: {str(e)}")
        return render_template('about.html', error="Error loading about page")

@main.route('/health')
def health():
    """Health check endpoint"""
    try:
        model_path = current_app.config['MODEL_PATH']
        logger.info(f"Health check - Model path: {model_path}")
        
        if os.path.exists(model_path):
            contents = os.listdir(model_path)
            logger.info(f"Health check - Directory contents: {contents}")
        else:
            logger.error(f"Health check - Model path does not exist: {model_path}")
            return jsonify({
                'status': 'error',
                'error': f'Model path does not exist: {model_path}'
            }), 500

        available_models = current_app.model_manager.get_available_models()
        logger.info(f"Health check - Available models: {available_models}")
        
        return jsonify({
            'status': 'healthy' if available_models else 'degraded',
            'model_path': model_path,
            'directory_contents': contents if os.path.exists(model_path) else [],
            'available_models': available_models
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
