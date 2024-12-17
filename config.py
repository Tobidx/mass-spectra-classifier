import os

class Config:
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Preprocessing settings
    ALIGNMENT_POINTS = 1000  # Number of points for spectrum alignment
    
    # Logging configuration
    LOG_PATH = os.path.join('logs')
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    
    # Development vs Production settings
    DEBUG = os.environ.get('DEBUG') == '1'
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    
# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
