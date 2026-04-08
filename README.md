# Real-Time Outfit Detection and Color Recommendation System

This system uses YOLOv8 for object detection, Hugging Face models for fine-grained color classification, and Ollama (local LLM) for intelligent fashion recommendations.

## 🚀 Features
- **Real-Time Webcam Feed**: Enhanced using spatial-domain processing.
- **Image Enhancement**: CLAHE, Gaussian Blur, Sobel edges, and Saturation boosting.
- **YOLOv8 Detection**: Specifically tracks clothing items and people.
- **AI Color Classification**: Uses `prithivMLmods/Fashion-Product-baseColour` (SigLIP2) for accurate color naming.
- **Ollama Recommendations**: Local LLM provides stylish advice based on your current outfit.

## 🛠 Installation

1. **Install Python Dependencies**:
   ```bash
   pip install opencv-python numpy ultralytics torch torchvision transformers pillow ollama requests
   ```

2. **Ollama Setup**:
   Ensure you have [Ollama](https://ollama.com/) installed and run:
   ```bash
   ollama pull llama3
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

## 🎮 Controls
- **Q**: Quit application
- **E**: Toggle Image Enhancement (On/Off)

## 📁 Project Structure
- `main.py`: Main application loop.
- `enhancement.py`: Phase 1 Image Processing logic.
- `detection.py`: Phase 2 YOLOv8 Integration.
- `color_utils.py`: Phase 3 AI Color classification.
- `recommender.py`: Phase 4 Ollama recommendation logic.
