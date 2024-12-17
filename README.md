# **Mass Spectrometry Classifier**

A machine learning-powered web application for mass spectra classification with **97.89% accuracy** and real-time prediction. This application streamlines the analysis of mass spectra data, enabling researchers and scientists to classify spectra efficiently.

[**Live Demo**](https://mass-spectrometry-classifier-application-289208564214.us-central1.run.app/)

---

## **Features**
- **High Accuracy**: Achieves 97.89% accuracy using optimized machine learning models.
- **Efficient Spectra Comparison**: Utilizes advanced ML algorithms (Random Forest, XGBoost) for accurate classification.
- **Real-Time Analysis**: Provides predictions in under 500ms.
- **Web-Based Interface**: Accessible through any browser with a user-friendly design.
- **Cloud Deployment**: Deployed on Google Cloud Run for scalability and reliability.
- **Dockerized Application**: Containerized for easy deployment and reproducibility.

---

## **Tech Stack**
- **Backend**: Python (Flask)
- **Frontend**: HTML, CSS
- **Machine Learning**: Random Forest, XGBoost
- **Containerization**: Docker
- **Deployment**: Google Cloud Run

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
   python app.py

5. **Access the App**:
   Open your browser and go to ```http://127.0.0.1:5000/.
   
---

## **Docker Setup**
To run the application using Docker:

1. **Build the Docker Image**:
   ```bash
   docker build -t mass-spectrometry-classifier .

2. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 mass-spectrometry-classifier

3. **Access the App**:
   Open your browser and go to ```http://127.0.0.1:5000/.


---

## **Deployment**
The application is deployed using Google Cloud Run. To deploy your own version:

1. Install and configure the Google Cloud SDK.

2. Build and push the Docker image to Google Container Registry (GCR):
   ```bash
   gcloud builds submit --tag gcr.io/mass-spectra-app/mass-spectrometry-classifier

3. Deploy to Google Cloud Run:
   ```bash
  gcloud run deploy --image gcr.io/mass-spectra-app/mass-spectrometry-classifier --platform managed

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


