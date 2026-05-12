import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

STATEMENT_MAP = {
    1: "Kids are talking by the door.",
    2: "Dogs are sitting by the door.",
}


@dataclass
class Config:
    root_dir: Path
    output_dir: Path
    sample_rate: int = 22050
    n_mels: int = 64
    max_audio_frames: int = 128
    vocab_size: int = 5000
    max_text_len: int = 20
    batch_size: int = 16
    epochs: int = 50
    test_actor_count: int = 5
    split_actor_count_stage_2: int = 6
    split_actor_count_stage_3: int = 3


def parse_ravdess_filename(path: Path) -> Dict[str, int]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return {
        "modality": int(parts[0]),
        "vocal_channel": int(parts[1]),
        "emotion": int(parts[2]),
        "intensity": int(parts[3]),
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor": int(parts[6]),
    }


def collect_dataset(root_dir: Path) -> pd.DataFrame:
    # Find speech (modality=03) files only to avoid duplicates
    wav_files = sorted(root_dir.rglob("03-*.wav"))
    if not wav_files:
        # Fallback: try broader search
        wav_files = sorted(set(root_dir.rglob("**/03-*.wav")))
    rows = []
    for wav in wav_files:
        try:
            meta = parse_ravdess_filename(wav)
            rows.append(
                {
                    "path": str(wav),
                    "emotion_id": meta["emotion"],
                    "emotion": EMOTION_MAP[meta["emotion"]],
                    "actor_id": meta["actor"],
                    "statement_id": meta["statement"],
                }
            )
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No .wav files found. Check dataset path.")
    return df


def actor_wise_split(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    actors = sorted(df["actor_id"].unique().tolist())
    trainval_actors, test_actors = train_test_split(
        actors,
        test_size=cfg.test_actor_count,
        random_state=SEED,
        shuffle=True,
    )
    train_actors, stage2_actors = train_test_split(
        trainval_actors,
        test_size=cfg.split_actor_count_stage_2,
        random_state=SEED,
        shuffle=True,
    )
    val_actors, fusion_actors = train_test_split(
        stage2_actors,
        test_size=cfg.split_actor_count_stage_3,
        random_state=SEED,
        shuffle=True,
    )

    splits = {
        "train": df[df["actor_id"].isin(train_actors)].reset_index(drop=True),
        "val": df[df["actor_id"].isin(val_actors)].reset_index(drop=True),
        "fusion_val": df[df["actor_id"].isin(fusion_actors)].reset_index(drop=True),
        "test": df[df["actor_id"].isin(test_actors)].reset_index(drop=True),
    }
    return splits


def augment_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    y_aug = y.copy()
    if random.random() < 0.7:
        y_aug = y_aug + 0.005 * np.random.randn(len(y_aug))
    if random.random() < 0.4:
        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=2)
    if random.random() < 0.4:
        y_aug = librosa.effects.time_stretch(y_aug, rate=1.2)
    return y_aug


def waveform_to_mel(y: np.ndarray, sr: int, n_mels: int, max_frames: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_frames:
        pad_width = max_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :max_frames]
    return mel_db.astype(np.float32)


def build_audio_tensor(df: pd.DataFrame, cfg: Config, augment: bool = False) -> np.ndarray:
    feats = []
    for path in df["path"]:
        y, sr = librosa.load(path, sr=cfg.sample_rate)
        if augment:
            y = augment_waveform(y, sr)
        mel = waveform_to_mel(y, sr, cfg.n_mels, cfg.max_audio_frames)
        feats.append(mel)
    x = np.stack(feats, axis=0)
    return x


def normalize_audio(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_fusion_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean()
    std = x_train.std() + 1e-8

    def _norm(x: np.ndarray) -> np.ndarray:
        z = (x - mean) / std
        return np.expand_dims(z, axis=-1)

    return _norm(x_train), _norm(x_val), _norm(x_fusion_val), _norm(x_test)


def init_whisper_model() -> Optional[object]:
    try:
        import whisper

        return whisper.load_model("tiny")
    except Exception:
        return None


def transcribe_audio(path: str, statement_id: int, whisper_model: Optional[object]) -> str:
    if whisper_model is not None:
        try:
            out = whisper_model.transcribe(path, language="en", fp16=False)
            text = out.get("text", "").strip()
            if text:
                return " ".join(text.lower().split())
        except Exception:
            pass
    return STATEMENT_MAP.get(statement_id, "")


def build_transcripts(
    df: pd.DataFrame,
    whisper_model: Optional[object],
    cache_file: Path,
) -> List[str]:
    cache: Dict[str, str] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    transcripts = []
    changed = False
    for _, row in df.iterrows():
        path = row["path"]
        if path in cache:
            transcripts.append(cache[path])
            continue
        txt = transcribe_audio(path, int(row["statement_id"]), whisper_model)
        cache[path] = txt
        transcripts.append(txt)
        changed = True

    if changed:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return transcripts


def export_transcripts_csv(df: pd.DataFrame, transcripts: List[str], output_csv: Path) -> None:
    # Keep transcript export compatible with the reference repository format.
    tmp = df.copy().reset_index(drop=True)
    tmp["transcript"] = transcripts
    tmp["clean_transcript"] = tmp["transcript"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    le = LabelEncoder()
    tmp["label"] = le.fit_transform(tmp["emotion"])

    tmp["path"] = tmp["path"].astype(str).str.replace("/", "\\", regex=False)
    out = tmp[["path", "emotion", "label", "transcript", "clean_transcript"]]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


def encode_text(
    train_texts: List[str],
    val_texts: List[str],
    fusion_texts: List[str],
    test_texts: List[str],
    vocab_size: int,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    def _encode(texts: List[str]) -> np.ndarray:
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    return (
        _encode(train_texts),
        _encode(val_texts),
        _encode(fusion_texts),
        _encode(test_texts),
        tokenizer,
    )


def labels_to_one_hot(y: np.ndarray, num_classes: int = 8) -> np.ndarray:
    # Emotion ids are 1..8 in RAVDESS.
    return to_categorical(y - 1, num_classes=num_classes)


def build_audio_model(input_shape: Tuple[int, int, int], num_classes: int = 8) -> Model:
    inp = Input(shape=input_shape, name="audio_input")

    x = Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    emb = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="audio_embedding")(x)
    x = Dropout(0.4)(emb)
    out = Dense(num_classes, activation="softmax", name="audio_output")(x)

    return Model(inp, out, name="audio_cnn")


def build_text_model(vocab_size: int, max_len: int, num_classes: int = 8) -> Model:
    inp = Input(shape=(max_len,), name="text_input")
    x = Embedding(input_dim=vocab_size, output_dim=64)(inp)
    x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
    emb = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="text_embedding")(x)
    x = Dropout(0.3)(emb)
    out = Dense(num_classes, activation="softmax", name="text_output")(x)
    return Model(inp, out, name="text_lstm")


def build_fusion_head(audio_dim: int = 128, text_dim: int = 64, num_classes: int = 8) -> Model:
    audio_in = Input(shape=(audio_dim,), name="audio_emb")
    text_in = Input(shape=(text_dim,), name="text_emb")
    x = Concatenate()([audio_in, text_in])
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model([audio_in, text_in], out, name="early_fusion_head")


def standard_callbacks() -> List[object]:
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str], title: str) -> Dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{title}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    return {"accuracy": acc, "macro_f1": macro_f1}


def plot_training_curves(history: tf.keras.callbacks.History, out_prefix: Path) -> None:
    hist = history.history

    plt.figure(figsize=(7, 4))
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_acc.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_loss.png"))
    plt.close()


def main() -> None:
    cfg = Config(
        root_dir=Path("task 4") / "archive",
        output_dir=Path("task 4") / "outputs",
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting dataset...")
    df = collect_dataset(cfg.root_dir)
    print(f"Total samples: {len(df)}")

    splits = actor_wise_split(df, cfg)
    for k, part in splits.items():
        print(f"{k}: {len(part)}")

    y_train = splits["train"]["emotion_id"].values
    y_val = splits["val"]["emotion_id"].values
    y_fusion_val = splits["fusion_val"]["emotion_id"].values
    y_test = splits["test"]["emotion_id"].values

    print("Building audio tensors...")
    x_train_audio = build_audio_tensor(splits["train"], cfg, augment=True)
    x_val_audio = build_audio_tensor(splits["val"], cfg, augment=False)
    x_fusion_audio = build_audio_tensor(splits["fusion_val"], cfg, augment=False)
    x_test_audio = build_audio_tensor(splits["test"], cfg, augment=False)
    x_train_audio, x_val_audio, x_fusion_audio, x_test_audio = normalize_audio(
        x_train_audio, x_val_audio, x_fusion_audio, x_test_audio
    )

    print("Preparing text transcripts...")
    whisper_model = init_whisper_model()
    if whisper_model is None:
        print("Whisper not available. Falling back to statement-based transcripts.")

    cache_file = cfg.output_dir / "transcript_cache.json"
    train_texts = build_transcripts(splits["train"], whisper_model, cache_file)
    val_texts = build_transcripts(splits["val"], whisper_model, cache_file)
    fusion_texts = build_transcripts(splits["fusion_val"], whisper_model, cache_file)
    test_texts = build_transcripts(splits["test"], whisper_model, cache_file)

    all_texts = build_transcripts(df, whisper_model, cache_file)
    export_transcripts_csv(df, all_texts, cfg.output_dir.parent / "ravdess_transcripts.csv")

    x_train_text, x_val_text, x_fusion_text, x_test_text, _ = encode_text(
        train_texts,
        val_texts,
        fusion_texts,
        test_texts,
        cfg.vocab_size,
        cfg.max_text_len,
    )

    y_train_oh = labels_to_one_hot(y_train)
    y_val_oh = labels_to_one_hot(y_val)
    y_fusion_oh = labels_to_one_hot(y_fusion_val)
    y_test_oh = labels_to_one_hot(y_test)

    label_names = [EMOTION_MAP[i] for i in range(1, 9)]

    print("Training Audio CNN...")
    audio_model = build_audio_model((cfg.n_mels, cfg.max_audio_frames, 1))
    audio_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train - 1),
        y=y_train - 1,
    )
    class_weights = {i: w for i, w in enumerate(class_weights_values)}
    h_audio = audio_model.fit(
        x_train_audio,
        y_train_oh,
        validation_data=(x_val_audio, y_val_oh),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=standard_callbacks(),
        class_weight=class_weights,
        verbose=1,
    )
    plot_training_curves(h_audio, cfg.output_dir / "cnn")

    print("Training Text LSTM...")
    text_model = build_text_model(cfg.vocab_size, cfg.max_text_len)
    text_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    h_text = text_model.fit(
        x_train_text,
        y_train_oh,
        validation_data=(x_val_text, y_val_oh),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=standard_callbacks(),
        verbose=1,
    )
    plot_training_curves(h_text, cfg.output_dir / "lstm")

    print("Extracting embeddings for fusion...")
    audio_embedder = Model(audio_model.input, audio_model.get_layer("audio_embedding").output)
    text_embedder = Model(text_model.input, text_model.get_layer("text_embedding").output)

    train_audio_emb = audio_embedder.predict(x_train_audio, verbose=0)
    fusion_audio_emb = audio_embedder.predict(x_fusion_audio, verbose=0)
    test_audio_emb = audio_embedder.predict(x_test_audio, verbose=0)

    train_text_emb = text_embedder.predict(x_train_text, verbose=0)
    fusion_text_emb = text_embedder.predict(x_fusion_text, verbose=0)
    test_text_emb = text_embedder.predict(x_test_text, verbose=0)

    print("Training Early Fusion head...")
    fusion_model = build_fusion_head(audio_dim=128, text_dim=64)
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h_fusion = fusion_model.fit(
        [train_audio_emb, train_text_emb],
        y_train_oh,
        validation_data=([fusion_audio_emb, fusion_text_emb], y_fusion_oh),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=standard_callbacks(),
        verbose=1,
    )
    plot_training_curves(h_fusion, cfg.output_dir / "fusion")

    print("Evaluating all models on unseen speakers...")
    audio_probs = audio_model.predict(x_test_audio, verbose=0)
    text_probs = text_model.predict(x_test_text, verbose=0)
    fusion_probs = fusion_model.predict([test_audio_emb, test_text_emb], verbose=0)

    audio_metrics = evaluate_predictions(y_test - 1, audio_probs, label_names, "Audio CNN")
    plt.savefig(cfg.output_dir / "audio_confusion.png")
    plt.close()

    text_metrics = evaluate_predictions(y_test - 1, text_probs, label_names, "Text LSTM")
    plt.savefig(cfg.output_dir / "text_confusion.png")
    plt.close()

    fusion_metrics = evaluate_predictions(y_test - 1, fusion_probs, label_names, "Early Fusion")
    plt.savefig(cfg.output_dir / "fusion_confusion.png")
    plt.close()

    results = pd.DataFrame(
        [
            {"model": "Audio CNN", **audio_metrics},
            {"model": "Text LSTM", **text_metrics},
            {"model": "Early Fusion", **fusion_metrics},
        ]
    )
    results.to_csv(cfg.output_dir / "results_summary.csv", index=False)
    print("\nResults summary:")
    print(results)
    print(f"\nSaved outputs in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
