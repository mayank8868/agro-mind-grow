# AgroMind Grow

AgroMind Grow is a comprehensive smart agriculture platform designed to empower farmers with AI-driven insights, real-time data, and expert guidance. From detecting plant diseases with high precision to providing market prices and weather forecasts, AgroMind Grow serves as an all-in-one digital farming assistant.

## Key Features

### 1. AI Disease Prediction (Plant Doctor)
*   **Instant Diagnosis**: Upload a photo of a plant leaf, and our advanced AI model identifies diseases with high accuracy.
*   **Smart Validation**: The system intelligently detects if an uploaded image is not a plant (e.g., random objects, people) and rejects it to prevent false diagnoses. It uses a combination of color analysis and confidence scoring to ensure validity.
*   **Comprehensive Reports**: Get detailed breakdowns of Symptoms, Causes, and Treatments (Organic, Chemical, and Prevention tips).
*   **Healthy Rescue**: If the model is unsure but the plant looks healthy, it intelligently reassesses to avoid false alarms.

### 2. Smart Dashboard
A centralized hub giving you a quick overview of your farm's status, weather alerts, and quick access to all tools.

### 3. Weather Forecast
Real-time weather updates and forecasts to help you plan your farming activities effectively.

### 4. Market Prices
Stay updated with the latest market prices for various crops to maximize your profits.

### 5. Crop Calendar
Personalized farming schedules and timelines to ensure you plant and harvest at the perfect time.

### 6. Planning & Consultation
Access expert advice and planning tools to optimize your farm's yield and sustainability.

### 7. Knowledge Base
A rich library of farming guides, articles, and best practices.

## Technical Stack

### Frontend
*   **Framework**: React (v18) with TypeScript
*   **Build Tool**: Vite for high-performance development and building.
*   **Styling**: Tailwind CSS for modern, responsive design.
*   **UI Components**: Shadcn UI for accessible and consistent components.
*   **Icons**: Lucide React.
*   **Charts**: Recharts for data visualization.

### Backend
*   **Framework**: FastAPI for high-performance API endpoints.
*   **Machine Learning**: PyTorch & Torchvision.
*   **Image Processing**: Pillow (PIL) and NumPy.

## The AI Model

Our disease prediction engine is built on a state-of-the-art Deep Learning architecture.

*   **Architecture**: EfficientNet-B2 (Transfer Learning). We chose B2 for its excellent balance between accuracy and inference speed, making it suitable for real-time applications.
*   **Training**:
    *   **Epochs**: 30 (with Early Stopping to prevent overfitting).
    *   **Optimizer**: AdamW with OneCycleLR scheduler for optimal convergence.
    *   **Loss Function**: CrossEntropyLoss with Label Smoothing to improve generalization.
    *   **Augmentations**: We use advanced techniques like RandomResizedCrop, ColorJitter, RandomRotation, and RandomErasing to make the model robust against different lighting and angles.
*   **Dataset**: Trained on a comprehensive dataset covering 38 classes of healthy and diseased plants.

## Project Structure

```
agro-mind-grow/
├── src/
│   ├── components/
│   │   ├── ui/                 # Reusable UI components (Button, Card, etc.)
│   │   └── Navigation.tsx      # Main navigation bar
│   ├── pages/
│   │   ├── CropCalendar.tsx
│   │   ├── Equipment.tsx
│   │   ├── Index.tsx           # Dashboard
│   │   ├── KnowledgeBase.tsx
│   │   ├── MarketPrices.tsx
│   │   ├── NotFound.tsx
│   │   ├── PestControl.tsx     # Disease Prediction Logic & UI
│   │   ├── PlanningConsultation.tsx
│   │   └── Weather.tsx
│   ├── App.tsx                 # Main App component
│   ├── main.tsx                # Entry point
│   └── index.css               # Global styles (Tailwind)
├── backend/
│   ├── models/                 # Saved model weights
│   ├── api.py                  # FastAPI application & Inference logic
│   ├── train.py                # Model training script
│   ├── disease_database.py     # Database of symptoms/treatments
│   └── requirements.txt        # Python dependencies
├── public/                     # Static assets
├── index.html                  # HTML entry point
├── package.json                # Frontend dependencies
├── tsconfig.json               # TypeScript configuration
└── vite.config.ts              # Vite configuration
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites
*   Node.js (v16+)
*   Python (v3.8+)
*   Git
*   CUDA-enabled GPU (Optional, but recommended for training)

### 1. Clone the Repository

Open your terminal and run the following command to clone the project:

```bash
git clone https://github.com/mayank8868/agro-mind-grow.git
cd agro-mind-grow
```

### 2. Backend Setup

Navigate to the project directory and set up the Python environment.

**Create and Activate Virtual Environment:**

```powershell
# Windows (PowerShell)
cd backend
python -m venv venv
.\venv\Scripts\Activate
```

**Install Dependencies:**

```powershell
pip install -r requirements.txt
```

**Start the Backend Server:**

Make sure your virtual environment is activated, then run:

```powershell
python api.py
```
The API will start at `http://localhost:8000`.

*(Optional) Train the Model:*
If you wish to retrain the model:
```powershell
python train.py
```

### 3. Frontend Setup

Open a new terminal window (keep the backend running) and navigate to the project root.

**Install Dependencies:**

```powershell
npm install
```

**Start the Application:**

You can use the provided batch script for a quick start:

```powershell
.\start.bat
```

Or run the standard npm command:

```powershell
npm run dev
```

The application will be available at `http://localhost:8080`.

## How It Works

1.  **User Action**: You drag & drop a plant image onto the Disease Prediction page.
2.  **Frontend**: The React app validates the file type and sends it to the backend API.
3.  **Backend Validation**:
    *   **Color Check**: The API analyzes the image pixels. If it doesn't contain enough "plant colors" (green/brown) or has too much blue (sky/jeans), it's rejected.
    *   **Confidence Check**: The AI predicts the disease. If the confidence score is below 50%, the image is rejected as "unclear/invalid".
4.  **Prediction**: If valid, the model identifies the specific disease (or healthy status).
5.  **Response**: The backend retrieves detailed treatment info from `disease_database.py` and sends it back.
6.  **Display**: The frontend renders a detailed report with the diagnosis, confidence score, and actionable advice.

## Contribution

Feel free to fork this repository and submit pull requests. We welcome improvements to the dataset, model architecture, or UI!

---

**Built for Farmers.**
