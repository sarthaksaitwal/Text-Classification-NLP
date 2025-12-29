# Text Classification NLP Project

## Project Overview
This project focuses on text classification using Natural Language Processing (NLP) techniques. It includes data preprocessing, tokenization, model training, evaluation, and inference. The project is implemented in Python and uses an LSTM model for classification tasks.

## Project Structure
```
app.py                      # Main application entry point
requirements.txt            # Python dependencies
project_structure.txt       # Project structure description
data/
    train.csv               # Training dataset
    test.csv                # Testing dataset
saved_models/
    lstm_best_model.h5      # Pre-trained LSTM model
src/
    evaluation/             # Evaluation metrics
        metrics.py
    inference/              # Inference logic
        predict.py
    models/                 # Model definitions
        lstm_model.py
    preprocessing/          # Text preprocessing utilities
        text_cleaning.py
    tokenization/           # Tokenization logic
        tokenizer.py
    training/               # Training scripts
        train_lstm.py
    utils/                  # Utility functions
        dataset_loader.py
static/
    style.css               # Static CSS files
templates/
    index.html              # HTML templates
```

## Features
- **Data Preprocessing**: Cleaning and preparing text data for training.
- **Tokenization**: Converting text into numerical representations.
- **Model Training**: Training an LSTM model on the dataset.
- **Model Evaluation**: Evaluating the model's performance using various metrics.
- **Inference**: Making predictions on new data.
- **Web Interface**: A simple web interface for user interaction.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Text-Classification-NLP.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Text-Classification-NLP
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Train the Model**:
   Run the training script to train the LSTM model:
   ```bash
   python src/training/train_lstm.py
   ```

2. **Evaluate the Model**:
   Evaluate the trained model using the evaluation script:
   ```bash
   python src/evaluation/metrics.py
   ```

3. **Run the Application**:
   Start the web application:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Dependencies
- Python 3.x
- TensorFlow/Keras
- Flask
- Pandas
- NumPy

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments
- The datasets used in this project.
- Open-source libraries and frameworks.
