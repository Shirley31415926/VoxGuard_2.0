import {
  FaceLandmarker,
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

const PACKAGE_VERSION = "0.10.35";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${PACKAGE_VERSION}/wasm`;
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const PHASES = [
  {
    id: "face",
    title: "F — Smile and show your teeth",
    instruction: "Keep your face centred and hold a natural smile. The demo measures mouth-corner symmetry.",
    durationMs: 5000,
    warmupMs: 1000
  },
  {
    id: "arms",
    title: "A — Raise both arms",
    instruction: "Raise both arms to a similar height and hold them there until the countdown ends.",
    durationMs: 8500,
    warmupMs: 1800
  },
  {
    id: "speech",
    title: "S — Repeat the sentence",
    instruction: "Say clearly: “Today is a bright day and I feel well.” This version measures voice activity and timing, not slurring.",
    durationMs: 7500,
    warmupMs: 600
  }
];

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24]
];
const FACE_POINTS = [10, 152, 61, 291, 13, 14, 1, 33, 263];

const $ = (id) => document.getElementById(id);
const elements = {
  systemBadge: $("systemBadge"),
  startButton: $("startButton"),
  baselineButton: $("baselineButton"),
  checkButton: $("checkButton"),
  stopButton: $("stopButton"),
  privacyToggle: $("privacyToggle"),
  cameraFrame: $("cameraFrame"),
  camera: $("camera"),
  overlay: $("overlay"),
  cameraPlaceholder: $("cameraPlaceholder"),
  phaseTitle: $("phaseTitle"),
  phaseInstruction: $("phaseInstruction"),
  countdown: $("countdown"),
  faceDot: $("faceDot"),
  poseDot: $("poseDot"),
  speechDot: $("speechDot"),
  fpsLabel: $("fpsLabel"),
  faceCard: $("faceCard"),
  armsCard: $("armsCard"),
  speechCard: $("speechCard"),
  faceValue: $("faceValue"),
  faceStatus: $("faceStatus"),
  armsValue: $("armsValue"),
  armsStatus: $("armsStatus"),
  speechValue: $("speechValue"),
  speechStatus: $("speechStatus"),
  audioMeter: $("audioMeter"),
  resultCard: $("resultCard"),
  resultTitle: $("resultTitle"),
  resultText: $("resultText"),
  baselineState: $("baselineState"),
  toast: $("toast")
};

let faceLandmarker = null;
let poseLandmarker = null;
let mediaStream = null;
let audioContext = null;
let analyser = null;
let audioBuffer = null;
let animationFrameId = null;
let modelsReady = false;
let sensorsReady = false;
let latestFace = null;
let latestPose = null;
let latestAudio = { rms: 0, threshold: 0.018, speaking: false };
let noiseFloor = 0.006;
let lastInferenceAt = 0;
let lastVideoTime = -1;
let modelTurn = "face";
let framesThisSecond = 0;
let fpsWindowStart = performance.now();
let currentRun = null;
let toastTimer = null;

let baseline = null;
try {
  baseline = JSON.parse(localStorage.getItem("voxguard-fast-baseline") || "null");
} catch {
  baseline = null;
}
updateBaselineLabel();

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function showToast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function setSystemBadge(text, mode) {
  elements.systemBadge.textContent = text;
  elements.systemBadge.className = `badge badge-${mode}`;
}

function setResult(mode, title, text) {
  elements.resultCard.className = `result-card card result-${mode}`;
  elements.resultTitle.textContent = title;
  elements.resultText.textContent = text;
}

function setMetricCard(card, mode) {
  card.classList.remove("active", "warning", "danger", "ok");
  if (mode) card.classList.add(mode);
}

function updateBaselineLabel() {
  if (baseline) {
    const date = new Date(baseline.createdAt);
    elements.baselineState.textContent = `Baseline: saved ${date.toLocaleDateString()}`;
  } else {
    elements.baselineState.textContent = "Baseline: not recorded";
  }
}

async function createTaskWithFallback(TaskClass, vision, options) {
  try {
    return await TaskClass.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "GPU" }
    });
  } catch (gpuError) {
    console.warn("GPU delegate unavailable; falling back to CPU.", gpuError);
    return TaskClass.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" }
    });
  }
}

async function initialiseModels() {
  try {
    setSystemBadge("Loading Face model…", "loading");
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

    faceLandmarker = await createTaskWithFallback(FaceLandmarker, vision, {
      baseOptions: { modelAssetPath: FACE_MODEL },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false
    });

    setSystemBadge("Loading Pose model…", "loading");
    poseLandmarker = await createTaskWithFallback(PoseLandmarker, vision, {
      baseOptions: { modelAssetPath: POSE_MODEL },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
      outputSegmentationMasks: false
    });

    modelsReady = true;
    elements.startButton.disabled = false;
    setSystemBadge("Models ready", "ready");
  } catch (error) {
    console.error(error);
    setSystemBadge("Model loading failed", "error");
    elements.phaseTitle.textContent = "Unable to load MediaPipe models";
    elements.phaseInstruction.textContent = "Check the internet connection, then reload the page.";
    showToast("MediaPipe models could not be loaded.");
  }
}

function isSecureCameraContext() {
  return window.isSecureContext || ["localhost", "127.0.0.1"].includes(location.hostname);
}

function waitForVideoReady(video, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("playing", onReady);
      video.removeEventListener("error", onError);
    };
    const finish = () => {
      if (settled) return;
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        settled = true;
        cleanup();
        resolve();
      }
    };
    const onReady = () => finish();
    const onError = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(video.error || new Error("The camera video element reported an error."));
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Camera permission was granted, but Safari did not deliver playable video frames."));
    }, timeoutMs);

    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("canplay", onReady);
    video.addEventListener("playing", onReady);
    video.addEventListener("error", onError);
    finish();
  });
}

async function initialiseAudio(stream) {
  const audioTracks = stream.getAudioTracks();
  if (!audioTracks.length) return false;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return false;

  try {
    audioContext = new AudioContextClass();
    await audioContext.resume();
    const source = audioContext.createMediaStreamSource(new MediaStream(audioTracks));
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.2;
    source.connect(analyser);
    audioBuffer = new Float32Array(analyser.fftSize);
    return true;
  } catch (error) {
    console.warn("Microphone analysis unavailable; continuing with camera analysis.", error);
    if (audioContext) await audioContext.close().catch(() => {});
    audioContext = null;
    analyser = null;
    audioBuffer = null;
    return false;
  }
}

async function startSensors() {
  if (!modelsReady || sensorsReady) return;
  if (!isSecureCameraContext()) {
    showToast("Camera and microphone require HTTPS on iPhone.");
    setResult("warning", "HTTPS required", "Deploy this folder to GitHub Pages, Netlify or Vercel before opening it on an iPhone.");
    return;
  }

  elements.phaseTitle.textContent = "Starting camera…";
  elements.phaseInstruction.textContent = "Allow camera and microphone access, then keep this tab visible.";
  elements.startButton.disabled = true;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 720 },
        height: { ideal: 960 },
        frameRate: { ideal: 24, max: 30 }
      },
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    });

    elements.camera.muted = true;
    elements.camera.autoplay = true;
    elements.camera.playsInline = true;
    elements.camera.setAttribute("playsinline", "");
    elements.camera.setAttribute("webkit-playsinline", "");
    elements.camera.srcObject = mediaStream;

    const playPromise = elements.camera.play();
    if (playPromise && typeof playPromise.catch === "function") {
      await playPromise.catch((error) => {
        console.warn("Initial video.play() was rejected; waiting for metadata and retrying.", error);
      });
    }
    await waitForVideoReady(elements.camera);
    if (elements.camera.paused) await elements.camera.play();

    const audioReady = await initialiseAudio(mediaStream);

    sensorsReady = true;
    lastVideoTime = -1;
    lastInferenceAt = 0;
    latestFace = null;
    latestPose = null;
    elements.cameraPlaceholder.hidden = true;
    elements.cameraFrame.classList.add("camera-live");
    elements.startButton.disabled = true;
    elements.baselineButton.disabled = false;
    elements.checkButton.disabled = false;
    elements.stopButton.disabled = false;
    elements.faceDot.classList.add("live");
    elements.poseDot.classList.add("live");
    elements.speechDot.classList.toggle("live", audioReady);
    elements.phaseTitle.textContent = "Sensors active";
    elements.phaseInstruction.textContent = audioReady
      ? "Record a normal baseline or start the guided FAST check."
      : "Camera analysis is active. Microphone analysis is unavailable in this browser session.";
    setResult(
      audioReady ? "neutral" : "warning",
      audioReady ? "Ready for guided check" : "Camera ready; microphone unavailable",
      "Raw media is processed locally in this browser and is not uploaded by this demo."
    );
    resizeCanvas();
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    renderLoop();
  } catch (error) {
    console.error(error);
    mediaStream?.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    elements.camera.pause();
    elements.camera.srcObject = null;
    elements.cameraFrame.classList.remove("camera-live");
    elements.cameraPlaceholder.hidden = false;
    elements.startButton.disabled = !modelsReady;
    elements.phaseTitle.textContent = "Sensors could not start";
    elements.phaseInstruction.textContent = describeMediaError(error);
    setResult("warning", "Permission or device error", describeMediaError(error));
    showToast(describeMediaError(error));
  }
}

function describeMediaError(error) {
  const name = error?.name || "";
  if (name === "NotAllowedError") return "Camera or microphone permission was denied. Enable access in Safari website settings and try again.";
  if (name === "NotFoundError") return "No usable camera or microphone was found.";
  if (name === "NotReadableError") return "The camera or microphone may already be in use by another app.";
  return error?.message || "The camera or microphone could not be started.";
}

function stopSensors() {
  currentRun = null;
  sensorsReady = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  animationFrameId = null;
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  elements.camera.pause();
  elements.camera.srcObject = null;
  elements.cameraFrame.classList.remove("camera-live");
  if (audioContext) audioContext.close().catch(() => {});
  audioContext = null;
  analyser = null;
  audioBuffer = null;
  elements.cameraPlaceholder.hidden = false;
  elements.startButton.disabled = !modelsReady;
  elements.baselineButton.disabled = true;
  elements.checkButton.disabled = true;
  elements.stopButton.disabled = true;
  elements.faceDot.className = "status-dot";
  elements.poseDot.className = "status-dot";
  elements.speechDot.className = "status-dot";
  elements.phaseTitle.textContent = "Sensors stopped";
  elements.phaseInstruction.textContent = "Tap Start camera & microphone to continue.";
  elements.countdown.textContent = "—";
  clearOverlay();
}

function resizeCanvas() {
  const width = elements.camera.videoWidth || 720;
  const height = elements.camera.videoHeight || 960;
  if (elements.overlay.width !== width || elements.overlay.height !== height) {
    elements.overlay.width = width;
    elements.overlay.height = height;
  }
}

function clearOverlay() {
  const ctx = elements.overlay.getContext("2d");
  ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
}

function updateAudio() {
  if (!analyser || !audioBuffer) return;
  analyser.getFloatTimeDomainData(audioBuffer);
  let squareSum = 0;
  for (const sample of audioBuffer) squareSum += sample * sample;
  const rms = Math.sqrt(squareSum / audioBuffer.length);

  const threshold = Math.max(0.014, noiseFloor * 2.8);
  const speaking = rms > threshold;
  if (!speaking) noiseFloor = (noiseFloor * 0.985) + (rms * 0.015);
  latestAudio = { rms, threshold, speaking };

  const meter = clamp((rms / Math.max(threshold * 2.3, 0.04)) * 100, 0, 100);
  elements.audioMeter.style.width = `${meter}%`;
  elements.speechDot.classList.toggle("active", speaking);

  if (!currentRun || currentRun.phase.id !== "speech") {
    elements.speechValue.textContent = speaking ? "Voice" : "Quiet";
    elements.speechStatus.textContent = `Local audio level ${(rms * 100).toFixed(1)}%`;
  }
}

function extractFaceFeatures(result) {
  const landmarks = result?.faceLandmarks?.[0];
  if (!landmarks) return null;
  const top = landmarks[10];
  const chin = landmarks[152];
  const leftMouth = landmarks[61];
  const rightMouth = landmarks[291];
  const faceHeight = distance(top, chin);
  if (faceHeight < 1e-6) return null;

  const vertical = {
    x: (chin.x - top.x) / faceHeight,
    y: (chin.y - top.y) / faceHeight
  };
  const mouthVector = {
    x: rightMouth.x - leftMouth.x,
    y: rightMouth.y - leftMouth.y
  };
  const verticalDifference = Math.abs((mouthVector.x * vertical.x) + (mouthVector.y * vertical.y));
  const asymmetry = verticalDifference / faceHeight;

  return { landmarks, asymmetry };
}

function extractPoseFeatures(result) {
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) return null;
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];
  const visible = [leftShoulder, rightShoulder, leftWrist, rightWrist]
    .every((point) => (point.visibility ?? 1) > 0.45);
  if (!visible) return null;

  const shoulderWidth = distance(leftShoulder, rightShoulder);
  if (shoulderWidth < 0.02) return null;

  const leftRelative = (leftWrist.y - leftShoulder.y) / shoulderWidth;
  const rightRelative = (rightWrist.y - rightShoulder.y) / shoulderWidth;
  const levelDifference = Math.abs(leftWrist.y - rightWrist.y) / shoulderWidth;
  const armsRaised = leftRelative < 0.30 && rightRelative < 0.30;

  return { landmarks, leftRelative, rightRelative, levelDifference, armsRaised };
}

function drawOverlay() {
  const canvas = elements.overlay;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (latestPose?.landmarks) {
    ctx.strokeStyle = "rgba(61, 217, 208, 0.88)";
    ctx.lineWidth = Math.max(3, canvas.width / 240);
    for (const [startIndex, endIndex] of POSE_CONNECTIONS) {
      const start = latestPose.landmarks[startIndex];
      const end = latestPose.landmarks[endIndex];
      if ((start.visibility ?? 1) < 0.4 || (end.visibility ?? 1) < 0.4) continue;
      ctx.beginPath();
      ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
      ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
      ctx.stroke();
    }
    for (const index of [11, 12, 13, 14, 15, 16]) {
      const point = latestPose.landmarks[index];
      ctx.fillStyle = index >= 15 ? "#ffbd59" : "#3dd9d0";
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(4, canvas.width / 150), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (latestFace?.landmarks) {
    ctx.fillStyle = "rgba(91, 140, 255, 0.92)";
    for (const index of FACE_POINTS) {
      const point = latestFace.landmarks[index];
      ctx.beginPath();
      ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(2.5, canvas.width / 260), 0, Math.PI * 2);
      ctx.fill();
    }
    const left = latestFace.landmarks[61];
    const right = latestFace.landmarks[291];
    ctx.strokeStyle = "rgba(255, 189, 89, 0.95)";
    ctx.lineWidth = Math.max(3, canvas.width / 280);
    ctx.beginPath();
    ctx.moveTo(left.x * canvas.width, left.y * canvas.height);
    ctx.lineTo(right.x * canvas.width, right.y * canvas.height);
    ctx.stroke();
  }
}

function updateLiveMetrics() {
  if (latestFace) {
    elements.faceValue.textContent = `${(latestFace.asymmetry * 100).toFixed(1)}%`;
    elements.faceStatus.textContent = "Live mouth-corner vertical difference";
  } else {
    elements.faceValue.textContent = "—";
    elements.faceStatus.textContent = "Face not clearly visible";
  }

  if (latestPose) {
    elements.armsValue.textContent = `${(latestPose.levelDifference * 100).toFixed(0)}%`;
    elements.armsStatus.textContent = latestPose.armsRaised ? "Both wrists detected in raised position" : "Raise both arms for assessment";
  } else {
    elements.armsValue.textContent = "—";
    elements.armsStatus.textContent = "Upper body not clearly visible";
  }
}

function activePhaseId() {
  return currentRun?.phase?.id || null;
}

function shouldRunFaceModel() {
  const phase = activePhaseId();
  if (phase === "face") return true;
  if (phase === "arms") return false;
  if (phase === "speech") return modelTurn === "face";
  return modelTurn === "face";
}

function collectCurrentSample(now) {
  if (!currentRun) return;
  const elapsed = now - currentRun.phaseStartedAt;
  if (elapsed < currentRun.phase.warmupMs) return;

  if (currentRun.phase.id === "face" && latestFace) {
    currentRun.samples.face.push({ t: now, asymmetry: latestFace.asymmetry });
  }
  if (currentRun.phase.id === "arms" && latestPose) {
    currentRun.samples.arms.push({
      t: now,
      leftRelative: latestPose.leftRelative,
      rightRelative: latestPose.rightRelative,
      levelDifference: latestPose.levelDifference,
      armsRaised: latestPose.armsRaised
    });
  }
  if (currentRun.phase.id === "speech") {
    currentRun.samples.speech.push({ t: now, ...latestAudio });
  }
}

function renderLoop(now = performance.now()) {
  if (!sensorsReady) return;
  updateAudio();
  resizeCanvas();

  const videoHasNewFrame = elements.camera.readyState >= 2 && elements.camera.currentTime !== lastVideoTime;
  if (videoHasNewFrame && now - lastInferenceAt >= 145) {
    try {
      if (shouldRunFaceModel()) {
        const result = faceLandmarker.detectForVideo(elements.camera, now);
        latestFace = extractFaceFeatures(result);
        modelTurn = "pose";
      } else {
        const result = poseLandmarker.detectForVideo(elements.camera, now);
        latestPose = extractPoseFeatures(result);
        modelTurn = "face";
      }
      lastInferenceAt = now;
      lastVideoTime = elements.camera.currentTime;
      framesThisSecond += 1;
    } catch (error) {
      console.warn("Inference frame skipped", error);
    }
  }

  drawOverlay();
  updateLiveMetrics();
  updateRunState(now);
  collectCurrentSample(now);

  if (now - fpsWindowStart >= 1000) {
    elements.fpsLabel.textContent = `${framesThisSecond} AI fps`;
    framesThisSecond = 0;
    fpsWindowStart = now;
  }

  animationFrameId = requestAnimationFrame(renderLoop);
}

function startGuidedRun(mode) {
  if (!sensorsReady || currentRun) return;
  currentRun = {
    mode,
    phaseIndex: 0,
    phase: PHASES[0],
    phaseStartedAt: performance.now(),
    samples: { face: [], arms: [], speech: [] }
  };
  elements.baselineButton.disabled = true;
  elements.checkButton.disabled = true;
  elements.stopButton.disabled = true;
  setResult("neutral", mode === "baseline" ? "Recording normal baseline" : "Guided FAST check in progress", "Follow the on-screen instructions. The sequence runs automatically.");
  activatePhaseUI();
}

function activatePhaseUI() {
  const phase = currentRun.phase;
  elements.phaseTitle.textContent = phase.title;
  elements.phaseInstruction.textContent = phase.instruction;
  [elements.faceCard, elements.armsCard, elements.speechCard].forEach((card) => setMetricCard(card, null));
  const activeCard = phase.id === "face" ? elements.faceCard : phase.id === "arms" ? elements.armsCard : elements.speechCard;
  setMetricCard(activeCard, "active");
  elements.faceDot.classList.toggle("active", phase.id === "face");
  elements.poseDot.classList.toggle("active", phase.id === "arms");
}

function updateRunState(now) {
  if (!currentRun) return;
  const elapsed = now - currentRun.phaseStartedAt;
  const remaining = Math.max(0, currentRun.phase.durationMs - elapsed);
  elements.countdown.textContent = `${Math.ceil(remaining / 1000)}s`;
  if (remaining > 0) return;

  currentRun.phaseIndex += 1;
  if (currentRun.phaseIndex >= PHASES.length) {
    finishGuidedRun();
    return;
  }
  currentRun.phase = PHASES[currentRun.phaseIndex];
  currentRun.phaseStartedAt = now;
  activatePhaseUI();
}

function analyseFace(samples) {
  const values = samples.map((sample) => sample.asymmetry).filter(Number.isFinite);
  if (values.length < 5) return { assessable: false, label: "Unable to assess" };
  const value = median(values);
  const baselineValue = baseline?.faceAsymmetry ?? 0;
  const threshold = baseline ? Math.max(0.043, baselineValue + 0.024) : 0.050;
  return {
    assessable: true,
    value,
    threshold,
    flagged: value > threshold,
    label: value > threshold ? "Asymmetry above demo threshold" : "Within demo threshold"
  };
}

function analyseArms(samples) {
  const valid = samples.filter((sample) => Number.isFinite(sample.leftRelative) && Number.isFinite(sample.rightRelative));
  if (valid.length < 8) return { assessable: false, label: "Unable to assess" };

  const raisedRatio = valid.filter((sample) => sample.armsRaised).length / valid.length;
  if (raisedRatio < 0.42) return { assessable: false, label: "Both arms were not held up long enough" };

  const raised = valid.filter((sample) => sample.armsRaised);
  const windowSize = Math.max(2, Math.floor(raised.length * 0.22));
  const start = raised.slice(0, windowSize);
  const end = raised.slice(-windowSize);
  const leftDrift = mean(end.map((s) => s.leftRelative)) - mean(start.map((s) => s.leftRelative));
  const rightDrift = mean(end.map((s) => s.rightRelative)) - mean(start.map((s) => s.rightRelative));
  const drift = Math.max(leftDrift, rightDrift, 0);
  const levelDifference = median(raised.map((sample) => sample.levelDifference));
  const baselineLevel = baseline?.armLevelDifference ?? 0;
  const levelThreshold = baseline ? Math.max(0.32, baselineLevel + 0.17) : 0.38;
  const driftThreshold = baseline ? Math.max(0.10, (baseline.armDrift ?? 0) + 0.08) : 0.13;
  const flagged = levelDifference > levelThreshold || drift > driftThreshold;

  return {
    assessable: true,
    value: levelDifference,
    drift,
    flagged,
    label: flagged ? "Level difference or downward drift detected" : "No marked drift detected"
  };
}

function analyseSpeech(samples) {
  if (samples.length < 20) return { assessable: false, label: "Unable to assess" };
  const voiced = samples.filter((sample) => sample.speaking);
  const voiceRatio = voiced.length / samples.length;
  const sampleDuration = samples.length > 1 ? (samples.at(-1).t - samples[0].t) / 1000 : 0;
  const voicedDuration = sampleDuration * voiceRatio;

  let pauseCount = 0;
  let wasSpeaking = false;
  let quietStartedAt = null;
  for (const sample of samples) {
    if (sample.speaking) {
      if (wasSpeaking && quietStartedAt && sample.t - quietStartedAt > 350) pauseCount += 1;
      wasSpeaking = true;
      quietStartedAt = null;
    } else if (wasSpeaking && quietStartedAt === null) {
      quietStartedAt = sample.t;
    }
  }

  if (voicedDuration < 1.0) {
    return { assessable: false, voiceRatio, voicedDuration, pauseCount, label: "Too little speech was detected" };
  }

  let changed = false;
  if (baseline?.speech) {
    const lowVoice = voiceRatio < Math.max(0.12, baseline.speech.voiceRatio * 0.52);
    const extraPauses = pauseCount > baseline.speech.pauseCount + 2;
    changed = lowVoice || extraPauses;
  }

  return {
    assessable: true,
    voiceRatio,
    voicedDuration,
    pauseCount,
    changed,
    label: baseline
      ? (changed ? "Timing differs from saved baseline" : "Voice timing close to baseline")
      : "Voice detected; no slurring model is included"
  };
}

function finishGuidedRun() {
  const completedRun = currentRun;
  currentRun = null;
  elements.baselineButton.disabled = false;
  elements.checkButton.disabled = false;
  elements.stopButton.disabled = false;
  elements.countdown.textContent = "Done";
  elements.phaseTitle.textContent = "Guided sequence complete";
  elements.phaseInstruction.textContent = "Review the three signal summaries below.";
  elements.faceDot.classList.remove("active");
  elements.poseDot.classList.remove("active");

  const face = analyseFace(completedRun.samples.face);
  const arms = analyseArms(completedRun.samples.arms);
  const speech = analyseSpeech(completedRun.samples.speech);

  if (completedRun.mode === "baseline") {
    if (!face.assessable || !arms.assessable || !speech.assessable) {
      setResult("warning", "Baseline was incomplete", "Repeat the baseline and ensure your face, upper body and voice remain detectable throughout all three steps.");
      showToast("Baseline not saved: one or more steps were incomplete.");
      return;
    }
    baseline = {
      createdAt: new Date().toISOString(),
      faceAsymmetry: face.value,
      armLevelDifference: arms.value,
      armDrift: arms.drift,
      speech: {
        voiceRatio: speech.voiceRatio,
        voicedDuration: speech.voicedDuration,
        pauseCount: speech.pauseCount
      }
    };
    localStorage.setItem("voxguard-fast-baseline", JSON.stringify(baseline));
    updateBaselineLabel();
    setResult("ok", "Normal baseline saved", "Future checks will compare face, arm and speech-timing features with this device-local baseline.");
    setMetricCard(elements.faceCard, "ok");
    setMetricCard(elements.armsCard, "ok");
    setMetricCard(elements.speechCard, "ok");
    showToast("Baseline saved locally on this device.");
    return;
  }

  renderAssessment(face, arms, speech);
}

function renderAssessment(face, arms, speech) {
  if (face.assessable) {
    elements.faceValue.textContent = `${(face.value * 100).toFixed(1)}%`;
    elements.faceStatus.textContent = face.label;
    setMetricCard(elements.faceCard, face.flagged ? "danger" : "ok");
  } else {
    elements.faceValue.textContent = "—";
    elements.faceStatus.textContent = face.label;
    setMetricCard(elements.faceCard, "warning");
  }

  if (arms.assessable) {
    elements.armsValue.textContent = `${(arms.value * 100).toFixed(0)}%`;
    elements.armsStatus.textContent = `${arms.label}; drift ${(arms.drift * 100).toFixed(0)}%`;
    setMetricCard(elements.armsCard, arms.flagged ? "danger" : "ok");
  } else {
    elements.armsValue.textContent = "—";
    elements.armsStatus.textContent = arms.label;
    setMetricCard(elements.armsCard, "warning");
  }

  if (speech.assessable) {
    elements.speechValue.textContent = `${speech.voicedDuration.toFixed(1)}s`;
    elements.speechStatus.textContent = `${speech.label}; ${speech.pauseCount} pause(s)`;
    setMetricCard(elements.speechCard, speech.changed ? "warning" : "ok");
  } else {
    elements.speechValue.textContent = "—";
    elements.speechStatus.textContent = speech.label;
    setMetricCard(elements.speechCard, "warning");
  }

  const visualWarning = face.flagged || arms.flagged;
  const incomplete = !face.assessable || !arms.assessable || !speech.assessable;
  if (visualWarning) {
    setResult(
      "danger",
      "Possible FAST warning sign",
      "This heuristic demo detected a facial or arm-movement deviation. Do not wait for another AI check if symptoms are sudden: seek emergency medical assessment now."
    );
  } else if (speech.changed || incomplete) {
    setResult(
      "warning",
      "Repeat or escalate the assessment",
      "No marked visual warning was detected, but one step was incomplete or speech timing differed from baseline. This demo cannot rule out stroke."
    );
  } else {
    setResult(
      "ok",
      "No marked deviation detected by this demo",
      "Face and arm signals remained within the heuristic thresholds. This result does not rule out stroke or replace a clinical FAST assessment."
    );
  }
}

elements.startButton.addEventListener("click", startSensors);
elements.stopButton.addEventListener("click", stopSensors);
elements.baselineButton.addEventListener("click", () => startGuidedRun("baseline"));
elements.checkButton.addEventListener("click", () => startGuidedRun("check"));
elements.privacyToggle.addEventListener("change", () => {
  elements.cameraFrame.classList.toggle("privacy", elements.privacyToggle.checked);
});

window.addEventListener("pagehide", () => {
  mediaStream?.getTracks().forEach((track) => track.stop());
});

initialiseModels();
