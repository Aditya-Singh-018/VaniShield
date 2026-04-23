import whisper
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from pydantic import BaseModel, Field
from typing import List

warnings.filterwarnings("ignore")

# -----------------------------
# 1. LLM STRUCTURED SCHEMA 
# -----------------------------
class FraudAnalysis(BaseModel):
    fraud_score: float = Field(..., ge=0, le=1)
    risk_level: str
    detected_tactics: List[str]
    suspicious_entities: List[str]
    reasoning_chain: str
    recommended_action: str

# -----------------------------
# 2. INITIALIZATION
# -----------------------------
print("Loading Whisper AI (Base Model)...")
whisper_model = whisper.load_model("base")

# -----------------------------
# 3. ANALYSIS LOGIC
# -----------------------------
def analyze_transcript_logic(transcript: str):
    transcript_lower = transcript.lower()
    red_flags = ["urgent", "immediately", "bank", "password", "otp", "verify", "account", "money", "transfer"]
    entities_to_check = ["hdfc", "sbi", "bank", "anydesk", "security", "office", "wallet"]
    
    found_flags = [word for word in red_flags if word in transcript_lower]
    found_entities = [word.capitalize() for word in entities_to_check if word in transcript_lower]
    
    score = min(1.0, len(found_flags) * 0.2)
    status = "CRITICAL" if score > 0.7 else "SUSPICIOUS" if score > 0.4 else "SAFE"
    
    return FraudAnalysis(
        fraud_score=score,
        risk_level=status,
        detected_tactics=found_flags if found_flags else ["None"],
        suspicious_entities=found_entities if found_entities else ["None Mentioned"],
        reasoning_chain=f"Transcript flagged {len(found_flags)} red flags. Score: {score}.",
        recommended_action="Freeze Account" if status != "SAFE" else "Monitor"
    )

def get_audio_features(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    cqt_db = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    phase = np.angle(librosa.stft(y))
    return {"y": y, "sr": sr, "S_db": S_db, "mfcc": mfcc, "cqt_db": cqt_db, "chroma": chroma, "f0": f0, "phase": phase}

# -----------------------------
# 4. FINAL INTEGRATED PLOTTING LOGIC
# -----------------------------
def save_enhanced_plot(r_feat, f_feat, sr, title, filename, plot_type="spec"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#f8f9fa')
    
    COLOR_REAL = '#1f77b4' 
    COLOR_FAKE = '#d62728' 
    
    # Standard labels for consistency
    axes[0].set_title("REFERENCE (REAL)", fontsize=11, fontweight='bold')
    axes[1].set_title("SAMPLE (FAKE)", fontsize=11, fontweight='bold')
    axes[2].set_title("FORENSIC COMPARISON", fontsize=11, fontweight='bold')

    if plot_type == "wave":
        librosa.display.waveshow(r_feat, sr=sr, ax=axes[0], color=COLOR_REAL)
        axes[0].set_ylabel("Amplitude")
        librosa.display.waveshow(f_feat, sr=sr, ax=axes[1], color=COLOR_FAKE)
        axes[1].set_ylabel("Amplitude")
        
        min_len = min(len(r_feat), len(f_feat))
        axes[2].plot(r_feat[:min_len], color=COLOR_REAL, label='Real', alpha=0.4)
        axes[2].plot(f_feat[:min_len], color=COLOR_FAKE, label='Fake', alpha=0.4)
        axes[2].set_ylabel("Overlay Amplitude")
        axes[2].legend()

    elif plot_type == "spec":
        img1 = librosa.display.specshow(r_feat, sr=sr, x_axis='time', y_axis='linear', ax=axes[0], cmap='viridis')
        axes[0].set_ylabel("Frequency (Hz)")
        plt.colorbar(img1, ax=axes[0], format="%+2.0f dB")

        img2 = librosa.display.specshow(f_feat, sr=sr, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
        axes[1].set_ylabel("Frequency (Hz)")
        plt.colorbar(img2, ax=axes[1], format="%+2.0f dB")

        min_w = min(r_feat.shape[1], f_feat.shape[1])
        diff = r_feat[:,:min_w] - f_feat[:,:min_w]
        img3 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='coolwarm', 
                              extent=[0, min_w*(512/sr), 0, sr/2])
        axes[2].set_ylabel("Frequency (Hz)")
        plt.colorbar(img3, ax=axes[2], label="dB Variance")

    elif plot_type == "mfcc":
        librosa.display.specshow(r_feat, x_axis='time', ax=axes[0], cmap='bone')
        axes[0].set_ylabel("MFCC Index")
        librosa.display.specshow(f_feat, x_axis='time', ax=axes[1], cmap='bone')
        
        axes[2].plot(np.mean(r_feat, axis=1), color=COLOR_REAL, marker='o', label='Real')
        axes[2].plot(np.mean(f_feat, axis=1), color=COLOR_FAKE, marker='x', label='Fake')
        axes[2].set_ylabel("Mean Magnitude")
        axes[2].legend()

    elif plot_type == "pitch":
        # Pitch Tracking (F0) with Delta Plot
        axes[0].plot(r_feat, color=COLOR_REAL)
        axes[0].set_ylabel("Frequency (Hz)")
        axes[1].plot(f_feat, color=COLOR_FAKE)
        axes[1].set_ylabel("Frequency (Hz)")
        
        min_len = min(len(r_feat), len(f_feat))
        r_clean = np.nan_to_num(r_feat[:min_len])
        f_clean = np.nan_to_num(f_feat[:min_len])
        pitch_diff = r_clean - f_clean
        
        axes[2].plot(pitch_diff, color='purple', linewidth=1)
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[2].fill_between(range(len(pitch_diff)), pitch_diff, color='purple', alpha=0.2)
        axes[2].set_ylabel("Delta F0 (Hz)")

    elif plot_type == "phase":
        librosa.display.specshow(r_feat, ax=axes[0], cmap='twilight')
        librosa.display.specshow(f_feat, ax=axes[1], cmap='twilight')
        
        min_w = min(r_feat.shape[1], f_feat.shape[1])
        diff = r_feat[:,:min_w] - f_feat[:,:min_w]
        axes[2].imshow(diff, aspect='auto', cmap='twilight')
        axes[2].set_ylabel("Phase Variance")

    # Global Formatting
    plt.suptitle(f"VANISHIELD FORENSIC UNIT: {title.upper()}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    
    plt.savefig(f"forensic_{filename}.png", dpi=300, bbox_inches='tight')
    print(f"File Saved: forensic_{filename}.png")
    plt.close()

# -----------------------------
# 5. EXECUTION
# -----------------------------
if __name__ == "__main__":
    REAL, FAKE = "samples/real2.wav", "samples/fake2.wav"

    if os.path.exists(REAL) and os.path.exists(FAKE):
        print("Transcribing and Extracting Features...")
        transcript = whisper_model.transcribe(FAKE)["text"]
        analysis = analyze_transcript_logic(transcript)
        
        r = get_audio_features(REAL)
        f = get_audio_features(FAKE)

        # Generate the full forensic suite
        save_enhanced_plot(r['y'], f['y'], r['sr'], "Waveform Analysis", "waveform", "wave")
        save_enhanced_plot(r['S_db'], f['S_db'], r['sr'], "Spectrogram Analysis", "spectrogram", "spec")
        save_enhanced_plot(r['mfcc'], f['mfcc'], r['sr'], "Vocal Texture (MFCC)", "mfcc", "mfcc")
        save_enhanced_plot(r['cqt_db'], f['cqt_db'], r['sr'], "Harmonic Profile (CQT)", "cqt", "spec")
        save_enhanced_plot(r['f0'], f['f0'], r['sr'], "Pitch Tracking (F0)", "pitch", "pitch")
        save_enhanced_plot(r['phase'], f['phase'], r['sr'], "Phase Discontinuity", "phase", "phase")
        
        print(f"\n--- Forensic Analysis Complete ---")
        print(f"Risk Level: {analysis.risk_level} | Fraud Score: {analysis.fraud_score}")
        print(f"Reasoning: {analysis.reasoning_chain}")
    else:
        print("Error: Missing samples/real2.wav or samples/fake2.wav")