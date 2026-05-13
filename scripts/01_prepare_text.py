import os
import re
import json
import nltk

# Scarica i tokenizzatori necessari se non presenti
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/synthetic_chunks"

def clean_text(text):
    # Rimuovi a capo singoli e unisci le parole sillabate
    # "parola-\ncapo" -> "parolacapo"
    text = re.sub(r'-\n\s*', '', text)
    # Sostituisci i restanti a capo con spazi
    text = re.sub(r'\n', ' ', text)
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_sentences(sentences, min_words=15, max_words=35, max_chars=200):
    """
    Hybrid Chunking:
    Raggruppa le frasi in modo che il numero di parole sia compreso tra min_words e max_words
    e che il testo non superi max_chars (limite di sicurezza per XTTS_v2 italiano = 213 chars).
    Se una singola frase è troppo lunga, viene spezzata sulle virgole o sui punti e virgola.
    """
    
    def split_long_sentence(sent, limit):
        """Spezza una frase lunga su virgole/punti e virgola."""
        if len(sent) <= limit:
            return [sent]
        # Prova a spezzare su virgola o punto e virgola
        parts = []
        current = ""
        for segment in re.split(r'(,\s|;\s)', sent):
            if len(current) + len(segment) > limit and current:
                parts.append(current.strip().rstrip(',').rstrip(';').strip())
                current = segment
            else:
                current += segment
        if current.strip():
            parts.append(current.strip().rstrip(',').rstrip(';').strip())
        return [p for p in parts if p]
    
    # Prima: spezza le frasi singole che superano il limite di caratteri
    expanded_sentences = []
    for sent in sentences:
        expanded_sentences.extend(split_long_sentence(sent, max_chars))
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sent in expanded_sentences:
        words = sent.split()
        word_count = len(words)
        candidate = " ".join(current_chunk + [sent])
        
        if (current_word_count + word_count > max_words or len(candidate) > max_chars) and current_chunk:
            # Salva il chunk corrente
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_word_count = word_count
        else:
            # Aggiungi al chunk corrente
            current_chunk.append(sent)
            current_word_count += word_count
            
        # Se abbiamo raggiunto il minimo, proviamo a chiudere il chunk
        if current_word_count >= min_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []
    
    # Processa tutti i file txt nella cartella raw
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        # 1. Pulizia testuale
        cleaned_text = clean_text(raw_text)
        
        # 2. Suddivisione in frasi (Sentence Tokenization)
        sentences = sent_tokenize(cleaned_text, language='italian')
        
        # 3. Hybrid Chunking
        chunks = chunk_sentences(sentences, min_words=15, max_words=35)
        
        base_name = os.path.splitext(filename)[0]
        for i, chunk in enumerate(chunks):
            chunk_id = f"{base_name}_chunk_{i:04d}"
            manifest.append({
                "id": chunk_id,
                "text": chunk,
                "source_file": filename
            })
            
    manifest_path = os.path.join(OUTPUT_DIR, "manifest_text.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
        
    print(f"Text prep completata. Generati {len(manifest)} chunk testuali.")
    print(f"Manifest salvato in: {manifest_path}")

if __name__ == "__main__":
    main()
