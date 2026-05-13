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
        
    print("Inizializzazione del modello XTTS_v2 in corso...")
    # XTTS_v2 supporta solo CUDA e CPU (MPS non è supportato dalla libreria Coqui TTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"Modello XTTS_v2 caricato su device: {device}")
    
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
        
        tts.tts_to_file(
            text=text, 
            speaker_wav=SPEAKER_WAV, 
            language="it", 
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
