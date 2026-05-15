import os
import json
import time
import torch

# Fix per PyTorch 2.6+: bypassa il blocco di sicurezza weights_only che fa crashare Coqui TTS
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Fix per torchaudio 2.5+: bypassa torchcodec (rimosso dalle dipendenze) usando soundfile
import torchaudio
import soundfile as sf

_original_torchaudio_load = torchaudio.load
def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Carica audio con soundfile invece di torchcodec."""
    data, sample_rate = sf.read(filepath, dtype="float32")
    tensor = torch.tensor(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # (1, samples) — mono
    else:
        tensor = tensor.T  # (channels, samples)
    return tensor, sample_rate
torchaudio.load = _patched_torchaudio_load

from TTS.api import TTS

TEXT_MANIFEST = "data/synthetic_chunks/manifest_text.json"
AUDIO_OUTPUT_DIR = "data/synthetic_audio"
FINAL_MANIFEST = "data/synthetic_audio/manifest_synthetic.json"
SPEAKER_WAV = "data/raw/reference_voice.wav"

def main():
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(TEXT_MANIFEST):
        print(f"Errore: manifest testuale {TEXT_MANIFEST} non trovato. Esegui prima 01_prepare_text.py.")
        return
        
    with open(TEXT_MANIFEST, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)

    # Seleziona il modello TTS in base alla disponibilità del file di riferimento vocale
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cloning = os.path.exists(SPEAKER_WAV)

    if use_cloning:
        print(f"Inizializzazione XTTS_v2 (voice cloning da {SPEAKER_WAV})...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"Modello XTTS_v2 caricato su device: {device}")
    else:
        print("⚠️  reference_voice.wav non trovato — uso voce italiana predefinita (VITS).")
        tts = TTS("tts_models/it/mai_female/vits").to(device)
        print(f"Modello VITS italiano caricato su device: {device}")

    synthetic_manifest = []
    
    total = len(text_chunks)
    print(f"Avvio generazione audio per {total} chunk...")
    
    for item in text_chunks:
        chunk_id = item['id']
        text = item['text']
        out_filename = f"{chunk_id}.wav"
        out_filepath = os.path.join(AUDIO_OUTPUT_DIR, out_filename)
        
        # Salta chunk già generati (utile per riprendere in caso di interruzione)
        if os.path.exists(out_filepath):
            synthetic_manifest.append({"id": chunk_id, "text": text, "audio_filepath": out_filepath})
            continue
        
        idx = text_chunks.index(item) + 1
        print(f"\n[{idx}/{total}] [{chunk_id}] Testo: {text[:80]}...")
        t0 = time.time()
        
        if use_cloning:
            tts.tts_to_file(
                text=text, 
                speaker_wav=SPEAKER_WAV, 
                language="it", 
                file_path=out_filepath
            )
        else:
            tts.tts_to_file(
                text=text,
                file_path=out_filepath
            )
        
        duration = time.time() - t0
        print(f"Fatto in {duration:.2f} secondi.")
        
        synthetic_manifest.append({
            "id": chunk_id,
            "text": text,
            "audio_filepath": out_filepath
        })
        
    with open(FINAL_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(synthetic_manifest, f, indent=4, ensure_ascii=False)
        
    print(f"\nGenerazione completata. Manifest finale salvato in {FINAL_MANIFEST}")

if __name__ == "__main__":
    main()
