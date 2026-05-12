# Multimodal Speech Emotion Recognition

# Project overview
This project focuses on 8-class speech emotion recognition using both audio and text from the RAVDESS dataset.

Emotion classes: angry, calm, disgust, fearful, happy, neutral, sad, surprised.

The system uses a multimodal approach:
- Audio branch: mel spectrograms processed using a CNN.
- Text branch: transcripts processed using an LSTM.
- Fusion stage: early fusion by concatenating latent embeddings before final classification.

# Dataset
The project uses speech audio from RAVDESS (24 actors, 1440 clips).

Key characteristics:
- Multiple speakers with different vocal styles.
- Fixed sentence content, which makes emotion more dependent on prosody than lexical variation.

## Split strategy
Actor-wise split is used to avoid speaker leakage.

- Train: actor-disjoint training split.
- Model validation: actor-disjoint validation split.
- Fusion validation: separate actor-disjoint split for fusion training validation.
- Test: unseen speakers only.

This setup measures generalization to unseen speakers rather than memorization of speaker identity.

# Audio pipeline
Audio preprocessing:
- Load waveform with librosa at sr=22050.
- Convert to mel spectrogram with n_mels=64.
- Convert to dB scale.
- Pad or truncate time axis to MAX_LEN=128.
- Normalize with train split mean and std.
- Expand channel dimension to shape (64, 128, 1).

Audio augmentation (train only):
- Gaussian noise.
- Pitch shift.
- Time stretch.

# Audio CNN arch
Input (64 x 128 x 1)
-> Conv2D(32, 3x3, ReLU, same, L2=1e-4)
-> BatchNormalization
-> MaxPooling2D(2x2)

-> Conv2D(64, 3x3, ReLU, same, L2=1e-4)
-> BatchNormalization
-> MaxPooling2D(2x2)

-> Conv2D(128, 3x3, ReLU, same, L2=1e-4)
-> BatchNormalization
-> MaxPooling2D(2x2)

-> GlobalAveragePooling2D
-> Dense(128, ReLU, L2=1e-4)
-> Dropout(0.4)
-> Dense(8, Softmax)

Reasoning:
- CNN captures local time-frequency patterns from spectrograms.
- Batch norm stabilizes optimization.
- GAP reduces parameter count and overfitting.
- Dropout and L2 improve generalization.

# Text pipeline
Text features are generated via transcription of each audio clip.

Implementation notes:
- Primary path: Whisper tiny transcription.
- Fallback path: statement-based transcript mapping when Whisper is unavailable.
- Lowercase plus whitespace normalization.
- Tokenizer with vocab size 5000 and OOV token.
- Tokenizer fit only on train transcripts.
- Integer sequences padded to length 20.

# Text LSTM arch
Input tokens (length=20)
-> Embedding(input_dim=5000, output_dim=64)
-> LSTM(64, dropout=0.2, recurrent_dropout=0.2)
-> Dense(64, ReLU, L2=1e-4)
-> Dropout(0.3)
-> Dense(8, Softmax)

Reasoning:
- Embedding maps tokens to dense semantic vectors.
- LSTM captures sequence context.
- Dense plus dropout regularizes compact text representation.

# Fusion arch
Audio embedding (128)
+
Text embedding (64)
-> Concatenation (192)
-> Dense(128, ReLU, L2=1e-4)
-> Dropout(0.3)
-> Dense(64, ReLU, L2=1e-4)
-> Dropout(0.3)
-> Dense(8, Softmax)

Why early fusion:
- Learns cross-modal interactions in a shared latent space.
- Simple and computationally efficient compared to more complex fusion mechanisms.

# Training strategy
- Loss:
  - Audio/Fusion: categorical crossentropy.
  - Text: categorical crossentropy with label smoothing (0.05).
- Optimizer: Adam.
- Regularization: dropout plus L2.
- Callbacks:
  - EarlyStopping(monitor=val_loss, patience=10, restore_best_weights=True)
  - ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=4, min_lr=1e-6)
- Batch size / max epochs: 16 / 50.
- Imbalance handling: class weights for audio branch.

# Evaluation metrics
- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion matrix

Macro F1 is emphasized because it weights each emotion class equally.

# Output format
Generated outputs follow repo-like naming and structure:
- task 4/ravdess_transcripts.csv
- task 4/outputs/cnn_acc.png
- task 4/outputs/cnn_loss.png
- task 4/outputs/lstm_acc.png
- task 4/outputs/lstm_loss.png
- task 4/outputs/fusion_acc.png
- task 4/outputs/fusion_loss.png
- task 4/outputs/audio_confusion.png
- task 4/outputs/text_confusion.png
- task 4/outputs/fusion_confusion.png
- task 4/outputs/results_summary.csv

# Analysis and discussion template
Use this section in your submission write-up:
- Compare audio vs text vs fusion on Accuracy and Macro F1.
- Explain if text underperforms due to low lexical diversity in RAVDESS.
- Discuss whether fusion helps or plateaus depending on text branch informativeness.
- Highlight speaker-generalization behavior under actor-wise split.
