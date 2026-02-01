# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE EMOTION RECOGNITION TRAINING - FINAL FIX
# Using Your emotions-dataset
# ══════════════════════════════════════════════════════════════════════════════

import os
import torch
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR = "C:\\VCAI\\emotion\\model"
NUM_EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MAX_LENGTH = 5 * 16000  # 5 seconds at 16kHz

# Emotion labels based on your dataset structure
EMOTION_LABELS = {
    "angry": 0,
    "happy": 1,
    "hesitant": 2,
    "interested": 3,
    "neutral": 4
}

LABEL_TO_ID = EMOTION_LABELS
ID_TO_LABEL = {v: k for k, v in EMOTION_LABELS.items()}

print(f"🎯 Emotion Labels: {EMOTION_LABELS}")
print(f"📊 Number of classes: {len(EMOTION_LABELS)}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_file(file_path, target_sr=16000):
    """Load and preprocess audio file"""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_emotions_folder():
    """
    Find the Emotions folder in Kaggle.
    Returns the path directly to the Emotions folder (not the base path).
    """
    # Possible paths in Kaggle
    possible_paths = [
        Path(r"C:\Emotions\Emotions")
    ]
    
    print("\n🔍 Searching for Emotions folder...")
    
    for emotions_path in possible_paths:
        print(f"  Checking: {emotions_path}")
        if emotions_path.exists() and emotions_path.is_dir():
            # Check if this path has the emotion folders
            emotion_folders = ["Angry", "Happy", "Hesitant", "Interested", "Neutral"]
            found_folders = [f for f in emotion_folders if (emotions_path / f).exists()]
            
            if len(found_folders) >= 3:  # At least 3 emotion folders found
                print(f"  ✅ Found Emotions folder at: {emotions_path}")
                print(f"  📁 Emotion folders found: {found_folders}")
                return emotions_path
    
    # If not found, list what's actually there
    print("\n❌ Emotions folder not found in expected locations!")
    print("\nLet's see what's in /kaggle/input/emotions-dataset:")
    base_path = Path("/kaggle/input/emotions-dataset")
    if base_path.exists():
        for item in base_path.iterdir():
            print(f"  - {item.name} (dir: {item.is_dir()})")
            if item.is_dir() and item.name.lower() in ['emotions', 'emotion']:
                print(f"    ✓ This looks like the Emotions folder!")
                for subitem in list(item.iterdir())[:10]:
                    print(f"      - {subitem.name} (dir: {subitem.is_dir()})")
    
    raise FileNotFoundError(
        "Could not find the Emotions folder!\n"
        "Please check the dataset is properly attached in Kaggle."
    )

def load_dataset_from_emotions_folder(emotions_path):
    """
    Load dataset directly from the Emotions folder.
    
    Args:
        emotions_path: Path to the Emotions folder (e.g., /kaggle/input/emotions-dataset/Emotions)
    """
    data = []
    
    print(f"\n📂 Loading data from: {emotions_path}")
    
    # Emotion folder mapping - the folder names should match what's in the filesystem
    emotion_folders = {
        "Angry": "angry",
        "Happy": "happy",
        "Hesitant": "hesitant",
        "Interested": "interested",
        "Neutral": "neutral"
    }
    
    for folder_name, emotion_label in emotion_folders.items():
        emotion_folder = emotions_path / folder_name
        
        if not emotion_folder.exists():
            print(f"⚠️  Folder not found: {emotion_folder}")
            # Try lowercase
            emotion_folder_lower = emotions_path / folder_name.lower()
            if emotion_folder_lower.exists():
                print(f"   Found lowercase version: {emotion_folder_lower}")
                emotion_folder = emotion_folder_lower
            else:
                continue
        
        # Find all audio files
        audio_files = []
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.MP3", "*.m4a", "*.M4A", "*.flac", "*.FLAC"]:
            found = list(emotion_folder.glob(ext))
            audio_files.extend(found)
            if found:
                print(f"    Found {len(found)} {ext} files")
        
        print(f"  ✅ {emotion_label:12s}: {len(audio_files):4d} files")
        
        # Add to dataset
        for audio_file in audio_files:
            data.append({
                "path": str(audio_file),
                "emotion": emotion_label,
                "label": EMOTION_LABELS[emotion_label]
            })
    
    print(f"\n✅ Total samples loaded: {len(data)}")
    
    if len(data) == 0:
        print("\n" + "="*80)
        print("ERROR: No audio files found!")
        print("="*80)
        print(f"Emotions path: {emotions_path}")
        print(f"Emotions path exists: {emotions_path.exists()}")
        print("\nContents of Emotions folder:")
        if emotions_path.exists():
            for item in emotions_path.iterdir():
                print(f"  - {item.name} (is_dir: {item.is_dir()})")
                if item.is_dir():
                    files = list(item.glob("*"))
                    print(f"    Contains {len(files)} items")
                    # Show first few files
                    for f in list(files)[:3]:
                        print(f"      - {f.name}")
        print("="*80)
        raise ValueError("No audio files found! Check the dataset structure above.")
    
    # Print distribution
    print("\n📊 Dataset distribution:")
    for emotion, label in EMOTION_LABELS.items():
        count = sum(1 for d in data if d["label"] == label)
        percentage = (count / len(data)) * 100 if len(data) > 0 else 0
        print(f"  {emotion:12s}: {count:4d} samples ({percentage:.1f}%)")
    
    return data

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

# Find the Emotions folder
emotions_folder = find_emotions_folder()

# Load all data - pass the Emotions folder directly
all_data = load_dataset_from_emotions_folder(emotions_folder)

# Split into train/val/test
print("\n" + "="*80)
print("SPLITTING DATASET")
print("="*80)

train_data, temp_data = train_test_split(
    all_data, 
    test_size=0.3, 
    random_state=42, 
    stratify=[d["label"] for d in all_data]
)

val_data, test_data = train_test_split(
    temp_data, 
    test_size=0.5, 
    random_state=42, 
    stratify=[d["label"] for d in temp_data]
)

print(f"\n📊 Dataset splits:")
print(f"  Train:      {len(train_data):4d} samples ({len(train_data)/len(all_data)*100:.1f}%)")
print(f"  Validation: {len(val_data):4d} samples ({len(val_data)/len(all_data)*100:.1f}%)")
print(f"  Test:       {len(test_data):4d} samples ({len(test_data)/len(all_data)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING FEATURE EXTRACTOR")
print("="*80)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    """Process audio files for the model"""
    audio_arrays = []
    
    for path in examples["path"]:
        audio = load_audio_file(path)
        if audio is not None:
            # Pad or truncate to MAX_LENGTH
            if len(audio) > MAX_LENGTH:
                audio = audio[:MAX_LENGTH]
            else:
                audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
            audio_arrays.append(audio)
        else:
            # If file fails to load, use silence
            audio_arrays.append(np.zeros(MAX_LENGTH))
    
    # Extract features
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    
    return {
        "input_values": inputs.input_values,
        "labels": examples["label"]
    }

# Create HuggingFace datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Apply preprocessing
print("\n" + "="*80)
print("PREPROCESSING DATASETS")
print("="*80)
print("🔄 This may take a few minutes...")

dataset_dict = dataset_dict.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=["path", "emotion"]
)

print("✅ Preprocessing complete!")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTION_LABELS),
    label2id=LABEL_TO_ID,
    id2label=ID_TO_LABEL,
    ignore_mismatched_sizes=True
)

# Freeze feature extractor
model.freeze_feature_encoder()

print(f"\n🤖 Model loaded: {MODEL_NAME}")
print(f"📊 Number of parameters: {model.num_parameters():,}")
print(f"🔒 Feature encoder frozen: Yes")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": acc,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    fp16=True,  # Use mixed precision for faster training
    save_total_limit=2,
    report_to=["tensorboard"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    compute_metrics=compute_metrics,
)

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"⏱️  Epochs: {NUM_EPOCHS}")
print(f"📦 Batch size: {BATCH_SIZE}")
print(f"📈 Learning rate: {LEARNING_RATE}")
print(f"💾 Output directory: {OUTPUT_DIR}")
print(f"🎯 Metric for best model: accuracy")

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN THE MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print("🚀 Training in progress...")

trainer.train()

print("\n✅ Training complete!")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

test_results = trainer.evaluate(dataset_dict["test"])

print("\n" + "="*80)
print("FINAL TEST RESULTS")
print("="*80)
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

final_model_path = f"{OUTPUT_DIR}/final"
trainer.save_model(final_model_path)
feature_extractor.save_pretrained(final_model_path)

print(f"\n💾 Model saved to: {final_model_path}")
print("\n🎉 All done! Your emotion recognition model is ready!")
print("\n" + "="*80)
