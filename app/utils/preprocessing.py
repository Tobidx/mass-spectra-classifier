import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import logging
import io
import csv

logger = logging.getLogger(__name__)

def load_spectrum(file):
    """Load mass spectrum from uploaded file with specific header format"""
    try:
        # Read file content
        content = file.read()
        file.seek(0)
        
        logger.debug(f"Processing file: {file.filename}")
        
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Split into lines
        lines = content.split('\n')
        logger.debug(f"Total lines in file: {len(lines)}")
        
        # Process the CSV using csv module
        mz_values = []
        intensity_values = []
        
        csv_reader = csv.reader(io.StringIO(content))
        next(csv_reader)  # Skip the first metadata line
        headers = next(csv_reader)  # Get the header line
        logger.debug(f"Headers: {headers}")
        
        # Process data rows
        for row in csv_reader:
            logger.debug(f"Processing row: {row}")
            if not row:  # Skip empty rows
                continue
            try:
                point = int(row[0])
                mz = float(row[1])
                intensity = float(row[2])
                mz_values.append(mz)
                intensity_values.append(intensity)
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping invalid row {row}: {str(e)}")
                continue
                
        logger.debug(f"Processed {len(mz_values)} data points")
        
        if not mz_values or not intensity_values:
            logger.error("No valid data points found")
            return None, None
            
        # Convert to numpy arrays
        mz = np.array(mz_values)
        intensities = np.array(intensity_values)
        
        # Basic validation
        if len(mz) == 0:
            logger.error("Empty mz array")
            return None, None
            
        if len(intensities) == 0:
            logger.error("Empty intensities array")
            return None, None
            
        # Normalize intensities
        max_intensity = np.max(intensities)
        if max_intensity > 0:
            intensities = intensities / max_intensity
            
        logger.debug(f"Final data - mz range: {np.min(mz):.2f} to {np.max(mz):.2f}")
        logger.debug(f"Final data - intensity range: {np.min(intensities):.2f} to {np.max(intensities):.2f}")
        
        return mz, intensities
        
    except Exception as e:
        logger.error(f"Error loading spectrum: {str(e)}")
        logger.error(f"Content preview: {content[:200] if content else 'No content'}")
        return None, None

def align_spectra(mz1, intensities1, mz2, intensities2, n_points=1000):
    """Align two spectra to a common m/z axis using interpolation"""
    try:
        # Find common m/z range
        min_mz = max(np.min(mz1), np.min(mz2))
        max_mz = min(np.max(mz1), np.max(mz2))
        
        logger.debug(f"Alignment range: {min_mz:.2f} to {max_mz:.2f}")
        
        common_mz = np.linspace(min_mz, max_mz, n_points)
        
        # Create interpolation functions
        f1 = interp1d(mz1, intensities1, kind='linear', bounds_error=False, fill_value=0)
        f2 = interp1d(mz2, intensities2, kind='linear', bounds_error=False, fill_value=0)
        
        # Interpolate intensities
        int1_aligned = f1(common_mz)
        int2_aligned = f2(common_mz)
        
        # Validate aligned data
        if np.any(np.isnan(int1_aligned)) or np.any(np.isnan(int2_aligned)):
            logger.error("NaN values in aligned intensities")
            return None, None, None
            
        return common_mz, int1_aligned, int2_aligned
        
    except Exception as e:
        logger.error(f"Error aligning spectra: {str(e)}")
        return None, None, None

def calculate_cosine_similarity(mz, int1, int2):
    """Calculate cosine similarity between aligned spectra"""
    try:
        # Input validation
        if len(int1) != len(int2):
            logger.error("Intensity arrays have different lengths")
            return None
            
        cosine_sim = np.dot(int1, int2) / (np.linalg.norm(int1) * np.linalg.norm(int2))
        
        if not np.isfinite(cosine_sim):
            logger.error("Invalid cosine similarity value")
            return None
            
        return {'cosine_similarity': float(cosine_sim)}
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return None

def calculate_top_features(mz, int1, int2):
    """Calculate all similarity features between spectra"""
    try:
        # Input validation
        if len(int1) != len(int2):
            logger.error("Intensity arrays have different lengths")
            return None
            
        # Calculate features
        cosine_sim = np.dot(int1, int2) / (np.linalg.norm(int1) * np.linalg.norm(int2))
        correlation = np.corrcoef(int1, int2)[0, 1]
        
        area1 = np.trapz(int1, mz)
        area2 = np.trapz(int2, mz)
        area_ratio = min(area1/area2, area2/area1) if area1 > 0 and area2 > 0 else 0
        
        features = {
            'cosine_similarity': float(cosine_sim),
            'correlation': float(correlation),
            'area_ratio': float(area_ratio)
        }
        
        # Validate features
        if not all(np.isfinite(v) for v in features.values()):
            logger.error("Invalid feature values calculated")
            return None
            
        logger.debug(f"Calculated features: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Error calculating features: {str(e)}")
        return None
