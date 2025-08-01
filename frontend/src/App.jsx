import React, { useState, useRef, useEffect, useCallback } from "react";
import { initializeApp } from "firebase/app";
import {
  getAuth,
  signInAnonymously,
  signInWithCustomToken,
  onAuthStateChanged,
} from "firebase/auth";
import {
  getFirestore,
  doc,
  getDoc,
  setDoc,
  collection,
  query,
  orderBy,
  limit,
  onSnapshot,
  serverTimestamp,
} from "firebase/firestore";

// Ensure __app_id, __firebase_config, and __initial_auth_token are available from the environment
const appId = typeof __app_id !== "undefined" ? __app_id : "default-app-id";
const firebaseConfig =
  typeof __firebase_config !== "undefined" ? JSON.parse(__firebase_config) : {};
const initialAuthToken =
  typeof __initial_auth_token !== "undefined" ? __initial_auth_token : null;

// Utility function to convert PCM audio data to WAV format
// This is included as a utility for potential future TTS features, not directly used for this app's current scope.
const pcmToWav = (pcmData, sampleRate) => {
  const pcm16 = pcmData; // Already Int16Array from API response
  const numChannels = 1; // Assuming mono audio
  const sampleRateHz = sampleRate;
  const bytesPerSample = 2; // 16-bit PCM

  const wavBuffer = new ArrayBuffer(44 + pcm16.length * bytesPerSample);
  const view = new DataView(wavBuffer);

  // RIFF chunk descriptor
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcm16.length * bytesPerSample, true); // ChunkSize
  writeString(view, 8, "WAVE");

  // FMT sub-chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true); // NumChannels
  view.setUint32(24, sampleRateHz, true); // SampleRate
  view.setUint32(28, sampleRateHz * numChannels * bytesPerSample, true); // ByteRate
  view.setUint16(32, numChannels * bytesPerSample, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample (16-bit)

  // DATA sub-chunk
  writeString(view, 36, "data");
  view.setUint32(40, pcm16.length * bytesPerSample, true); // Subchunk2Size

  // Write PCM data
  let offset = 44;
  for (let i = 0; i < pcm16.length; i++, offset += bytesPerSample) {
    view.setInt16(offset, pcm16[i], true);
  }

  return new Blob([view], { type: "audio/wav" });
};

const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

const base64ToArrayBuffer = (base64) => {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};

function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [predictedDigit, setPredictedDigit] = useState("?");
  const [predictionConfidence, setPredictionConfidence] = useState(0); // New state for confidence
  const [score, setScore] = useState(0);
  const [targetDigit, setTargetDigit] = useState(null);
  const [message, setMessage] = useState("");
  const [leaderboard, setLeaderboard] = useState([]);
  const [db, setDb] = useState(null);
  const [auth, setAuth] = useState(null);
  const [userId, setUserId] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [userName, setUserName] = useState("");
  const [showNameInput, setShowNameInput] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Initialize Firebase and Auth
  useEffect(() => {
    try {
      const app = initializeApp(firebaseConfig);
      const firestore = getFirestore(app);
      const authInstance = getAuth(app);
      setDb(firestore);
      setAuth(authInstance);

      onAuthStateChanged(authInstance, async (user) => {
        if (user) {
          setUserId(user.uid);
          // Try to get existing user name
          const userDocRef = doc(
            firestore,
            `artifacts/${appId}/users/${user.uid}/user_data/profile`
          );
          const userDocSnap = await getDoc(userDocRef);
          if (userDocSnap.exists() && userDocSnap.data().name) {
            setUserName(userDocSnap.data().name);
          } else {
            setShowNameInput(true); // Prompt for name if not found
          }
        } else {
          // Sign in anonymously if no user is logged in
          if (initialAuthToken) {
            await signInWithCustomToken(authInstance, initialAuthToken);
          } else {
            await signInAnonymously(authInstance);
          }
        }
        setIsAuthReady(true); // Auth state is now ready
      });
    } catch (error) {
      console.error("Error initializing Firebase:", error);
      setMessage("Error initializing game services. Please try again later.");
    }
  }, []);

  // Save user name to Firestore
  const saveUserName = async () => {
    if (db && userId && userName.trim() !== "") {
      try {
        const userDocRef = doc(
          db,
          `artifacts/${appId}/users/${userId}/user_data/profile`
        );
        await setDoc(
          userDocRef,
          { name: userName.trim(), userId: userId },
          { merge: true }
        );
        setShowNameInput(false);
        setMessage("Name saved!");
      } catch (error) {
        console.error("Error saving user name:", error);
        setMessage("Error saving name. Please try again.");
      }
    }
  };

  // Canvas setup
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 15; // Thicker line for digit drawing
    ctx.lineCap = "round";
    ctx.strokeStyle = "white"; // Draw in white on a black background

    // Set canvas dimensions dynamically for responsiveness
    const resizeCanvas = () => {
      const container = canvas.parentElement;
      canvas.width = container.clientWidth;
      canvas.height = container.clientWidth; // Make it square
      // Reapply drawing styles after resize
      ctx.lineWidth = 15;
      ctx.lineCap = "round";
      ctx.strokeStyle = "white";
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height); // Clear and set background
    };

    resizeCanvas(); // Initial resize
    window.addEventListener("resize", resizeCanvas); // Listen for window resize

    // Initial clear to black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
    };
  }, []);

  // Drawing functions
  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = nativeEvent;
    const ctx = canvasRef.current.getContext("2d");
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      predictDigit(canvas);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      setPredictedDigit("?");
      setPredictionConfidence(0); // Reset confidence
      setMessage("Draw a digit!");
    }
  };

  // Prediction function (NOW CONNECTED TO PYTHON BACKEND VIA NGROK)
  const predictDigit = async (canvas) => {
    setIsLoading(true);
    setMessage("Predicting...");
    setPredictionConfidence(0); // Clear previous confidence

    // Get the canvas content as a Data URL (base64 encoded PNG)
    const imageData = canvas.toDataURL("image/png");

    //  <-- PASTE YOUR NGROK HTTPS URL HERE -->
    //  This line must be updated manually every time Ngrok is restarted.
    const backendUrl = "http://127.0.0.1:5000";

    try {
      // Send the image data to your Flask backend via ngrok
      const response = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
      }

      const data = await response.json();
      const predicted = data.prediction;
      const confidence = data.confidence; // Get confidence from response

      setPredictedDigit(predicted);
      setPredictionConfidence(confidence * 100); // Convert to percentage

      // Check if prediction matches targetDigit
      if (targetDigit !== null) {
        if (predicted === targetDigit) {
          setScore((prev) => prev + 1);
          setMessage("Correct!");
        } else {
          setMessage(`Incorrect. It was ${targetDigit}.`);
        }
      } else {
        setMessage(`Predicted: ${predicted}`);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      setMessage(
        `Prediction failed: ${error.message}. Make sure your backend server is running and accessible.`
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Game Logic
  const startNewRound = useCallback(() => {
    clearCanvas();
    const newTarget = Math.floor(Math.random() * 10);
    setTargetDigit(newTarget);
    setMessage(`Draw the digit: ${newTarget}`);
    setPredictedDigit("?");
    setPredictionConfidence(0); // Reset confidence for new round
  }, []);

  useEffect(() => {
    startNewRound(); // Start the first round when component mounts
  }, [startNewRound]);

  // Save score to Firestore
  const saveScore = async () => {
    if (!db || !userId || !isAuthReady) {
      setMessage("Authentication not ready. Cannot save score.");
      return;
    }

    if (score === 0) {
      setMessage("Score is 0. Nothing to save.");
      return;
    }

    setIsLoading(true);
    try {
      // Store private user score
      const userScoreDocRef = doc(
        db,
        `artifacts/${appId}/users/${userId}/scores`,
        `score_${Date.now()}`
      );
      await setDoc(userScoreDocRef, {
        score: score,
        timestamp: serverTimestamp(),
        userId: userId,
        userName: userName || "Anonymous", // Use saved name or Anonymous
      });

      // Store public leaderboard score (only if it's a high score or for top N)
      // For simplicity, we'll just add it to a public collection.
      // In a real app, you'd check if it's a top score before adding to avoid too many docs.
      const leaderboardCollectionRef = collection(
        db,
        `artifacts/${appId}/public/data/leaderboard`
      );
      await setDoc(doc(leaderboardCollectionRef), {
        score: score,
        timestamp: serverTimestamp(),
        userId: userId,
        userName: userName || "Anonymous",
      });

      setMessage(`Score ${score} saved!`);
      setScore(0); // Reset score after saving
      startNewRound(); // Start a new round
    } catch (error) {
      console.error("Error saving score:", error);
      setMessage("Error saving score. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch leaderboard from Firestore
  useEffect(() => {
    if (!db || !isAuthReady) return;

    const leaderboardCollectionRef = collection(
      db,
      `artifacts/${appId}/public/data/leaderboard`
    );
    const q = query(
      leaderboardCollectionRef,
      orderBy("score", "desc"),
      limit(10)
    ); // Top 10 scores

    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        const scores = [];
        snapshot.forEach((doc) => {
          scores.push({ id: doc.id, ...doc.data() });
        });
        setLeaderboard(scores);
      },
      (error) => {
        console.error("Error fetching leaderboard:", error);
        setMessage("Error loading leaderboard.");
      }
    );

    return () => unsubscribe(); // Cleanup listener
  }, [db, isAuthReady]);

  return (
    <div className="min-h-screen bg-gray-900 text-white font-inter flex flex-col items-center justify-center p-4">
      <script src="https://cdn.tailwindcss.com"></script>
      <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
        rel="stylesheet"
      />

      {/* Name Input Modal */}
      {showNameInput && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-8 rounded-lg shadow-xl border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-center">
              Enter Your Name
            </h2>
            <input
              type="text"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              placeholder="Your Name"
              className="w-full p-3 mb-4 rounded-md bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={saveUserName}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-md transition duration-300 ease-in-out transform hover:scale-105"
            >
              Save Name
            </button>
          </div>
        </div>
      )}

      <h1 className="text-5xl font-extrabold mb-8 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600 animate-pulse">
        Digit Recognizer Game
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl">
        {/* Game Area */}
        <div className="md:col-span-2 bg-gray-800 p-6 rounded-lg shadow-xl border border-gray-700 flex flex-col items-center">
          <h2 className="text-3xl font-bold mb-4 text-center">
            Draw the Digit:{" "}
            <span className="text-yellow-400">{targetDigit}</span>
          </h2>
          <div className="relative w-full max-w-md aspect-square bg-black rounded-lg overflow-hidden border-2 border-gray-600">
            <canvas
              ref={canvasRef}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseOut={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
              className="w-full h-full cursor-crosshair"
            ></canvas>
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 text-white text-xl font-semibold">
                Predicting...
              </div>
            )}
          </div>
          <div className="flex flex-col sm:flex-row gap-4 mt-6 w-full justify-center">
            <button
              onClick={clearCanvas}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-md transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
            >
              Clear
            </button>
            <button
              onClick={startNewRound}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-md transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
            >
              New Round
            </button>
            <button
              onClick={saveScore}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-md transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
              disabled={isLoading || score === 0}
            >
              Save Score
            </button>
          </div>
        </div>

        {/* Info and Leaderboard */}
        <div className="md:col-span-1 flex flex-col gap-8">
          {/* Current Game Info */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl border border-gray-700">
            <h2 className="text-3xl font-bold mb-4 text-center">Game Info</h2>
            <p className="text-xl mb-2">
              Your Score:{" "}
              <span className="font-bold text-green-400">{score}</span>
            </p>
            <p className="text-xl mb-2">
              Predicted:{" "}
              <span className="font-bold text-purple-400">
                {predictedDigit}
              </span>
              {predictedDigit !== "?" && (
                <span className="ml-2 text-sm text-gray-400">
                  ({predictionConfidence.toFixed(2)}% confident)
                </span>
              )}
            </p>
            <p className="text-lg text-center mt-4 text-gray-300">{message}</p>
            {userId && (
              <p className="text-sm mt-2 text-gray-500 text-center break-all">
                Your User ID: {userId}
              </p>
            )}
            {userName && (
              <p className="text-md mt-2 text-gray-400 text-center">
                Welcome, {userName}!
                <button
                  onClick={() => setShowNameInput(true)}
                  className="ml-2 text-blue-400 hover:text-blue-300 text-sm"
                >
                  (Edit Name)
                </button>
              </p>
            )}
          </div>

          {/* Leaderboard */}
          <div className="bg-gray-800 p-6 rounded-lg shadow-xl border border-gray-700">
            <h2 className="text-3xl font-bold mb-4 text-center">Leaderboard</h2>
            {leaderboard.length === 0 ? (
              <p className="text-center text-gray-400">
                Loading leaderboard...
              </p>
            ) : (
              <ol className="list-decimal list-inside text-lg">
                {leaderboard.map((entry, index) => (
                  <li
                    key={entry.id}
                    className="flex justify-between items-center py-1 border-b border-gray-700 last:border-b-0"
                  >
                    <span className="font-semibold text-gray-200">
                      {index + 1}. {entry.userName || "Anonymous"}
                    </span>
                    <span className="font-bold text-yellow-300">
                      {entry.score}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;