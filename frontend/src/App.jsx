import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css'; 

function App() {
  // --- THE NEW SPLASH SCREEN STATE ---
  const [hasStarted, setHasStarted] = useState(false);

  const webcamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const requestRef = useRef(null);
  const progressRef = useRef(null); 
  
  const [prediction, setPrediction] = useState('Initializing Engine...');
  const [sentence, setSentence] = useState([]); 
  
  const lastGuess = useRef('Ready');
  const stableCount = useRef(0);

  // --- NATIVE WEB AUDIO SYNTHESIZER ---
  const playSound = (type) => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);
      const now = ctx.currentTime;

      if (type === 'add') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(400, now);
        osc.frequency.exponentialRampToValueAtTime(500, now + 0.1);
        gain.gain.setValueAtTime(0.1, now); 
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
        osc.start(now);
        osc.stop(now + 0.1);
      } else if (type === 'delete') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(300, now);
        osc.frequency.exponentialRampToValueAtTime(150, now + 0.15);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
        osc.start(now);
        osc.stop(now + 0.15);
      } else if (type === 'clear') {
        osc.type = 'square';
        osc.frequency.setValueAtTime(400, now);
        osc.frequency.exponentialRampToValueAtTime(50, now + 0.3);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
        osc.start(now);
        osc.stop(now + 0.3);
      }
    } catch (e) {
      console.log("Audio contextual block");
    }
  };

  useEffect(() => {
    // Only initialize the heavy AI models AFTER the user clicks start
    if (!hasStarted) return;

    const initializeAI = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU" 
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        landmarkerRef.current = handLandmarker;
        setPrediction('Ready');
      } catch (error) {
        console.error("AI Load Error:", error);
        setPrediction('Connection Error');
      }
    };
    initializeAI();
  }, [hasStarted]); // <--- Added dependency here

  const predictWebcam = () => {
    if (landmarkerRef.current && webcamRef.current && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const results = landmarkerRef.current.detectForVideo(video, Date.now());
      
      let sign = "Ready";

      if (results.landmarks && results.landmarks.length > 0) {
        const lms = results.landmarks[0]; 
        const getDist = (p1, p2) => Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2);
        
        const indexUp = getDist(lms[8], lms[0]) > getDist(lms[6], lms[0]);
        const middleUp = getDist(lms[12], lms[0]) > getDist(lms[10], lms[0]);
        const ringUp = getDist(lms[16], lms[0]) > getDist(lms[14], lms[0]);
        const pinkyUp = getDist(lms[20], lms[0]) > getDist(lms[18], lms[0]);
        const thumbUp = getDist(lms[4], lms[17]) > getDist(lms[3], lms[17]);

        const state = `${thumbUp ? 1 : 0}${indexUp ? 1 : 0}${middleUp ? 1 : 0}${ringUp ? 1 : 0}${pinkyUp ? 1 : 0}`;

        switch(state) {
          case "00000": sign = "CLOSED FIST"; break; 
          case "11000": sign = "TAKE THE L"; break;  
          case "11111": sign = "HELLO"; break;
          case "01000": sign = "THIS"; break;
          case "01100": sign = "IS"; break;
          case "01110": sign = "OUR"; break;
          case "01111": sign = "PROJECT"; break;
          case "01001": sign = "ROCK ON"; break;
          case "10001": sign = "CALL ME"; break;
          case "11001": sign = "I LOVE YOU"; break;
          default: sign = "Ready"; break;
        }
      }

      if (sign !== "Ready") {
        if (sign === lastGuess.current) {
          stableCount.current += 1;
          
          if (stableCount.current === 30) {
            if (sign === "TAKE THE L") {
              setSentence([]); 
              playSound('clear');
            } else if (sign === "CLOSED FIST") {
              setSentence(prev => prev.slice(0, -1)); 
              playSound('delete');
              stableCount.current = 0; 
            } else {
              setSentence(prev => [...prev, sign]); 
              playSound('add');
            }
          }
        } else {
          stableCount.current = 0; 
          lastGuess.current = sign;
        }
      } else {
        stableCount.current = 0; 
        lastGuess.current = "Ready";
      }

      if (progressRef.current) {
        let fillPercentage = stableCount.current / 30;
        if (fillPercentage > 1) fillPercentage = 1; 
        if (sign === "Ready") fillPercentage = 0; 
        progressRef.current.style.transform = `scaleX(${fillPercentage})`;
      }

      setPrediction(sign);
    }
    
    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  useEffect(() => {
    if (hasStarted) {
      requestRef.current = requestAnimationFrame(predictWebcam);
      return () => cancelAnimationFrame(requestRef.current);
    }
  }, [hasStarted]); // <--- Added dependency here

  // --- UTILITY FUNCTIONS ---
  const handleSpeak = () => {
    if (sentence.length > 0) {
      const textToSpeak = sentence.join(" ");
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = 0.9; 
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleCopy = () => {
    if (sentence.length > 0) {
      const textToCopy = sentence.join(" ");
      navigator.clipboard.writeText(textToCopy);
      const btn = document.getElementById("copyBtn");
      if (btn) {
        btn.innerHTML = "✓ COPIED";
        setTimeout(() => btn.innerHTML = "📋 COPY", 2000);
      }
    }
  };

  // ==========================================
  // RENDER PHASE
  // ==========================================

  // 1. If not started, show the Splash Screen
  if (!hasStarted) {
    return (
      <div className="sv-splash-container">
        <div className="sv-splash-content">
          <h1 className="sv-splash-title">SilentVoice</h1>
          <p className="sv-splash-subtitle">EDGE-COMPUTED TRANSLATION ENGINE</p>
          <button className="sv-start-btn" onClick={() => setHasStarted(true)}>
            <div className="pulse"></div>
            INITIALIZE SYSTEM
          </button>
        </div>
      </div>
    );
  }

  // 2. If started, show the main application
  return (
    <div className="sv-container">
      <nav className="sv-nav">
        <div className="sv-brand">SilentVoice <span className="sv-badge">Edge AI</span></div>
        <div className="sv-status"><div className="pulse"></div> Local Engine Active</div>
      </nav>

      <main className="sv-main">
        <div className="sv-panel">
          <div className="panel-header">Camera Feed</div>
          <div className="video-wrapper">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
              mirrored={true} 
            />
          </div>
        </div>

        <div className="sv-panel sv-data-panel">
          <div className="panel-header">Translation Engine</div>
          
          <div className="sv-raw-output">
            <span className="sv-label">Detected Gesture</span>
            <div className="sv-prediction">
              {prediction === "Ready" ? <span style={{opacity: 0.3}}>{prediction}</span> : prediction}
            </div>
            <div className="sv-progress-container">
              <div className="sv-progress-fill" ref={progressRef}></div>
            </div>
          </div>
          
          <div className="sv-sentence-output">
            <span className="sv-label">Constructed Sentence</span>
            <div className="sv-sentence-box">
              {sentence.length > 0 ? sentence.join(" ") : <span className="sv-placeholder">Start signing to build a sentence...</span>}
            </div>
            
            <div className="sv-utility-bar">
              <button className="sv-utility-btn" onClick={handleSpeak}>
                🔊 SPEAK
              </button>
              <button id="copyBtn" className="sv-utility-btn" onClick={handleCopy}>
                📋 COPY
              </button>
            </div>
          </div>

          <div className="sv-controls-legend">
            <div className="sv-legend-item"><kbd>FIST</kbd> Backspace</div>
            <div className="sv-legend-item"><kbd>L-SHAPE</kbd> Clear All</div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;