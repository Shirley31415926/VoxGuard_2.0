# VoxGuard FAST Mobile Demo

A static, browser-only demonstration of:

- **Face:** MediaPipe Face Landmarker and a tilt-normalised mouth-corner asymmetry feature.
- **Arms:** MediaPipe Pose Landmarker, wrist-level difference and downward arm drift.
- **Speech:** Web Audio API voice activity detection (VAD), voiced duration and pause timing.
- **Privacy mode:** hides the raw video while retaining landmark overlays.
- **Baseline:** optionally stores normal feature summaries in browser `localStorage`.

## Important limitation

This is a user-flow and technical feasibility prototype. It is **not a medical device**, does not diagnose or exclude stroke, and has not been clinically validated. The speech module detects voice activity and timing only; it does not detect slurring or aphasia.

## Why it works on iPhone

The app is a static web page and uses the phone browser's camera, microphone, Web Audio API and WebAssembly. It includes `playsinline` and requests access only after a user taps the Start button.

Camera and microphone access requires a secure context. On an iPhone, deploy the folder over **HTTPS**. Opening `index.html` directly from Files is not sufficient.

## Fastest deployment: GitHub Pages

1. Create a new GitHub repository.
2. Upload the contents of this folder to the repository root.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Choose `main` and `/ (root)`, then save.
6. Open the generated `https://<username>.github.io/<repository>/` address in Safari on the iPhone.
7. Tap **Start camera & microphone** and allow both permissions.

Netlify Drop or Vercel static deployment also works.

## Local desktop testing

Camera access works on `localhost`:

```bash
python3 -m http.server 8080
```

Then open:

```text
http://localhost:8080
```

Do not test by double-clicking `index.html`; browser module and permission rules can block it.

## Architecture

```text
Camera frame
  ├─ MediaPipe Face Landmarker → mouth-corner asymmetry
  └─ MediaPipe Pose Landmarker → wrist level + arm drift

Microphone
  └─ Web Audio API → RMS energy → adaptive VAD → voice duration + pauses

All outputs
  └─ heuristic guided FAST summary
```

There is no backend and no upload endpoint. MediaPipe JavaScript/WASM and model files are downloaded from Google/jsDelivr when the page loads.

## Recommended next technical steps

1. Replace heuristic thresholds with thresholds calibrated on consented study data.
2. Add a true acoustic speech model using log-mel spectrograms or embeddings.
3. Move synchronous MediaPipe inference into a Web Worker if older phones feel slow.
4. Add encrypted, consent-based event export only after a data-governance design is approved.
5. Test across older iPhones, lighting conditions, skin tones, accents and mobility impairments.


## Safari/iPhone preview fix in this build

This revision fixes two mobile-browser problems:

1. The placeholder layer previously used `display: grid`, which could override the HTML `hidden` attribute and cover a working camera feed.
2. Camera startup now waits for playable video frames, retries `video.play()`, and keeps Face/Arms analysis available even if Web Audio initialisation fails.

After replacing files on GitHub Pages, open the page in Safari and perform a hard refresh. If the previous build remains cached, append `?v=2` to the page URL once.
