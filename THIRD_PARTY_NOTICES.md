# Third-party components and model assets

Extract Stills runs analysis locally. Runtime extraction does not provision or download models. Model downloads are confined to the explicit developer/build command `python scripts/setup_models.py`.

## Bundled model assets

The authoritative versions, download URLs, sizes and SHA256 digests are in `models/manifest.json`. Model binaries remain outside Git history and are included in packaged releases after verification. No Paddle model conversion or Paddle framework is required: both text models are PaddlePaddle's official ONNX exports.

- **MediaPipe Face Landmarker, float16 version 1**. Copyright Google; Apache License 2.0. The bundle contains BlazeFace short-range detection, Face Mesh V2 and blendshape prediction. Each model card states Apache-2.0: [detector](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf), [mesh](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf), [blendshapes](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf). [Official source](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker#models).
- **PP-OCRv5 mobile text detector ONNX**. Copyright PaddlePaddle Authors; Apache License 2.0. [Pinned official model card](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/blob/e6f4fa85f00e168c862bc462aebca69eef9b3d3d/README.md); [license](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE). Preprocessing and DB thresholds follow that revision's `inference.yml`. This app implements rectangle expansion and ROI measurements itself.
- **PP-OCRv5 Latin mobile text recognizer ONNX and character dictionary**. Copyright PaddlePaddle Authors; Apache License 2.0. [Pinned official model card](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/blob/89d3a50e2c27e2e7cceeab0e944c25c807d5db4f/README.md); [matching inference configuration/dictionary](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/blob/89d3a50e2c27e2e7cceeab0e944c25c807d5db4f/inference.yml); [license](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE). The Latin model supports English and accented Spanish along with other Latin-script languages. The application implements native-frame perspective crops, BGR normalization and greedy CTC decoding, and loads only verified local assets with ONNX Runtime CPU. Character confidence alone does not establish that text is legible or that animated words are complete; selection also uses boundary and temporal stability evidence.

The full Apache-2.0 license must accompany release assets, along with applicable attribution/NOTICE material. It is available at https://www.apache.org/licenses/LICENSE-2.0.txt and in the distributions of MediaPipe and OpenCV.

## Runtime libraries

- MediaPipe 0.10.21: Apache-2.0, [source and license](https://github.com/google-ai-edge/mediapipe/tree/v0.10.21).
- ONNX Runtime CPU 1.23.2: MIT, [source and license](https://github.com/microsoft/onnxruntime/tree/v1.23.2).
- OpenCV contrib 4.11: Apache-2.0; its Python wheel contains additional notices for bundled native dependencies. Include the wheel's license directory. [OpenCV](https://github.com/opencv/opencv/tree/4.11.0).
- NumPy: BSD-3-Clause with notices for bundled numerical libraries. [License](https://numpy.org/doc/stable/license.html).
- PySceneDetect: BSD-3-Clause. [License](https://github.com/Breakthrough/PySceneDetect/blob/main/LICENSE).
- PySide6/Qt: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only, subject to its selected distribution and bundled component licenses. Include Qt/PySide notices in native releases. [Licensing](https://doc.qt.io/qtforpython-6/licenses.html).
- FFmpeg/FFprobe: the actual binary build controls the applicable LGPL/GPL license. Release builds must exclude nonfree components and distribute matching source, build configuration, licenses and external-library notices. Color export requires `libzimg` and `liblcms2`. [FFmpeg license and redistribution](https://ffmpeg.org/legal.html).
- MediaPipe transitive components, including JAX/JAXlib, protobuf, FlatBuffers, matplotlib, sounddevice and their native dependencies, retain their upstream licenses and notices. Release packaging must collect distribution license/NOTICE directories rather than treating this short inventory as their full license texts.

## Offline and privacy audit

MediaPipe newer releases document performance/usage telemetry in their [current privacy notice](https://github.com/google-ai-edge/mediapipe#privacy-notice). The application requires the pinned 0.10.21 CPU runtime; upgrading it requires another network audit. The absence of a privacy notice in an older source tree is not evidence of a complete binary audit.

Release acceptance must run face/text inference with networking unavailable and verify no attempted outbound connections from the packaged process. ONNX sessions explicitly use `CPUExecutionProvider`; Face Landmarker explicitly uses the CPU delegate and local verified assets. No API keys, identity recognition or external inference service is used. Face confidence is a geometric reliability estimate because Face Landmarker does not expose a per-face detector probability; small/cropped detections receive neutral expression scores.

Development smoke on Windows x64/Python 3.12.9 verified the pinned runtime against the two hashed assets, including a two-face image, a synthetic title, and FFmpeg-decoded frames from the supplied videos. A 10 ms Windows IPv4/IPv6 TCP-table poll observed no outbound TCP connections belonging to the analysis process during startup, 40 inference calls and shutdown; Python socket connection operations were simultaneously denied. This bounded observation is not a substitute for the native packaged Windows/macOS offline release gate, and does not prove absence of all possible network attempts.

Text-recognition regression checks additionally verify the two pinned Latin recognition assets, multiword English, accented Spanish, perspective/native crop handling, vertical and upside-down lines, severe blur, boundary-cut glyphs, CTC repeated characters and spaces, and failure on missing or tampered assets. Recognition and lazy ONNX loading are exercised with Python socket connection operations denied; complete packaged networking audits remain part of release acceptance.
