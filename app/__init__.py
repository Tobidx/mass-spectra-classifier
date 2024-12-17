from flask import Flask
from config import Config
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Set model path in config
    model_path = os.getenv('MODEL_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    app.config['MODEL_PATH'] = model_path
    
    # Debug log the model path and contents
    logger.info(f"Model path set to: {app.config['MODEL_PATH']}")
    if os.path.exists(app.config['MODEL_PATH']):
        logger.info(f"Model directory contents: {os.listdir(app.config['MODEL_PATH'])}")
    else:
        logger.error(f"Model directory does not exist: {app.config['MODEL_PATH']}")
    
    # Ensure upload directory exists
    os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)
    
    # Initialize ModelManager
    from app.models.model_manager import ModelManager
    try:
        app.model_manager = ModelManager(app.config['MODEL_PATH'])
        logger.info("ModelManager initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ModelManager: {str(e)}", exc_info=True)
        raise
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app
