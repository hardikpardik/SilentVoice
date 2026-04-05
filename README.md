# SilentVoice 

**Scalable Edge-Computed Sign Language Translation Engine**

SilentVoice is a zero-latency, offline-first web application that translates sign language into text and synthesized speech in real-time. By migrating spatial feature extraction and Neural Network classification entirely to the client-side edge via WebAssembly, this project eliminates the network latency, high hosting costs, and privacy vulnerabilities of traditional server-bound AI tools.

## 🚀 Features
* **Zero-Latency Inference:** Runs entirely in the browser at 60 FPS without making a single network request.
* **Edge-Native ML Pipeline:** Utilizes Google's MediaPipe compiled to WebAssembly (WASM) to extract 21 precise 3D hand coordinates.
* **Temporal Stabilization Buffer:** A custom 30-frame debounce engine prevents UI flickering and ensures only mathematically stable hand geometries are registered.
* **Multimodal Feedback:** Native Web Audio API integration for real-time auditory confirmation (synthesized oscillators) and Web Speech API for room-level text-to-speech.
* **Continuous Hardware States:** Supports continuous physical controls (e.g., holding a closed fist to continuously backspace).

## 🧠 System Architecture

The project is structured as a Monorepo, split into two distinct environments:

### 1. The Neural Backend (`/backend`)
A localized Python environment used strictly for data collection and model training. 
* Captures raw XYZ coordinates for thousands of hand landmarks.
* Trains a lightweight classification model to recognize spatial patterns over strict mathematical rules.
* Exports the compiled model (`action.h5`) for edge deployment.

### 2. The Edge Frontend (`/frontend`)
A React application optimized with Vite. 
* Acts as the presentation and execution layer.
* Ingests the optical feed, runs the WASM spatial extraction, and passes the normalized features to the ML classifier.
* Manages state, the sentence buffer, and the "Arctic Frost" glassmorphism UI.

## 💻 Local Installation

### Prerequisites
* Node.js (v18+)
* Python 3.9+ (For backend ML training only)

### Frontend Setup (Running the App)
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SilentVoice.git](https://github.com/YOUR_USERNAME/SilentVoice.git)
