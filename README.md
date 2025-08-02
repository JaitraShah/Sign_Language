# Sign Language Detector
This project is a real-time sign language recognition system that uses computer vision and deep learning to recognize ASL (A–Z) letters and custom sign language words (e.g., Hello, Thank You, Yes, No).
The system captures gestures from a webcam and translates them into text for easier communication between signers and non-signers.

Features
- Real-time hand gesture detection using MediaPipe
- Recognition of A–Z alphabets using the ASL Alphabet dataset
- Add support for custom words using your own data collection
- CNN-based classifier trained on 87,000+ images
- Live webcam prediction with displayed text

Tools & Libraries Used
- Python 3.12
- OpenCV – Webcam & image processing
- MediaPipe – Hand landmark detection
- TensorFlow/Keras – Deep learning model
- scikit-learn – Data splitting & evaluation
- ImageDataGenerator – Efficient dataset loading

How to Run
1. Clone the repository:
git clone https://github.com/<your-username>/sign-language-recognition.git
cd sign-language-recognition

2. Install dependencies
   pip install -r requirements.txt

3. Train the model (If not using pre-trained model): python train_model_asl.py
4. Run real-time prediction: python predict_sign.py
