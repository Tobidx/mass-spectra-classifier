# **Mass Spectrometry Classifier**

A machine learning-powered web application for mass spectra classification with **97.89% accuracy** and real-time prediction. Users can upload two mass spectra files and get instant analysis of their similarity, helping to determine if they represent the same compound.

[**Live Demo**](https://mass-spectrometry-classifier-application-289208564214.us-central1.run.app/)

---

## **Features**

- **Multiple Analysis Models**

- XGBoost with Cosine Similarity
- XGBoost with Top 3 Features
- Random Forest with Cosine Similarity
- Random Forest with Top 3 Features

- **Advanced Analysis Metrics**

- Cosine Similarity Analysis
- Peak Correlation Measurement
- Area Ratio Comparison
- Confidence Scoring

- **Interactive Visualization**

- Real-time Spectra Comparison
- Interactive Plot Manipulation
- Downloadable Results

---

## **Tech Stack**
- Backend Framework: Python/Flask
- Frontend: HTML5, TailwindCSS, JavaScript
- Machine Learning:XGBoost, Scikit-learn, NumPy/Pandas
- Visualization: Plotly.js
- Deployment: Google Cloud Run
- CI/CD: Google Cloud Build
  
---

## **Installation Instructions**
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/mass-spectrometry-classifier.git
   cd mass-spectrometry-classifier

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run the Application**:
   ```bash
   export FLASK_APP=run.py
   export FLASK_ENV=development
   flask run

5. **Access the App**:
   Open your browser and go to ```http://127.0.0.1:5000/.
   
---

## **Docker Setup**
To run the application using Docker:

1. **Build the Docker Image**:
   ```bash
   docker build -t mass-spectra-classifier .

2. **Run the Docker Container**:
   ```bash
   docker run -p 8080:8080 mass-spectra-classifier

3. **Access the App**:
   Open your browser and go to ```http://127.0.0.1:5000/.


---

## **Deployment**
The application is deployed using Google Cloud Run. To deploy your own version:

1. Install and configure the Google Cloud SDK.

2. Build and push the Docker image to Google Container Registry (GCR):
   ```bash
   gcloud config set project mass-spectra-app

3. Deploy to Google Cloud Run:
   ```bash
  gcloud builds submit --config cloudbuild.yaml

---

## **Usage**

1. Upload your mass spectra data file (in supported formats like .csv or .json).
2. The application processes the data and compares spectra.
3. Receive real-time classification results.
   
---

## **Performance**

- Accuracy: 97.89%
- Prediction Time: <500ms
- Models: Random Forest, XGBoost optimized for spectra classification tasks.


Made with ❤️ by Oluwatobiloba Ajibola

