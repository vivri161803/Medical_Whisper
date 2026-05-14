"""
03_data_contracts.py — Contratti Dati Pydantic per la pipeline di Fine-Tuning.

Valida la struttura del dataset in ingresso, garantendo:
- Sample rate a 16kHz
- Canali mono
- Durata ≤ 30 secondi
- Testo ripulito da tag HTML/XML
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Union

import soundfile as sf
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Modelli Pydantic
# ---------------------------------------------------------------------------

class ManifestEntry(BaseModel):
    """Singola entry del manifest: riferimento audio + trascrizione.

    Supporta i formati di campo usati sia dal manifest sintetico
    (audio_filepath, text) che dal manifest augmented (audio_path, transcript).
    """
    id: str
    audio_path: str = Field(
        ...,
        validation_alias=AliasChoices("audio_path", "audio_filepath"),
    )
    text: str = Field(
        ...,
        validation_alias=AliasChoices("text", "transcript"),
    )
    duration_sec: Optional[float] = None
    augmented_audio_path: Optional[str] = None

    model_config = {"populate_by_name": True}

    @field_validator("text")
    @classmethod
    def text_must_be_clean(cls, v: str) -> str:
        """Il testo non deve essere vuoto e non deve contenere tag HTML/XML."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Il testo della trascrizione è vuoto.")
        if re.search(r"<[^>]+>", stripped):
            raise ValueError(
                f"Il testo contiene tag HTML/XML non consentiti: {stripped[:80]}..."
            )
        return stripped


class AudioSample(BaseModel):
    """Risultato della validazione di un singolo campione audio."""
    id: str
    audio_path: str
    sample_rate: int
    channels: int
    duration_sec: float
    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)

    # Vincoli
    REQUIRED_SAMPLE_RATE: int = 16000
    REQUIRED_CHANNELS: int = 1
    MAX_DURATION_SEC: float = 30.0

    @model_validator(mode="after")
    def validate_audio_constraints(self) -> "AudioSample":
        new_errors = []
        if self.sample_rate != self.REQUIRED_SAMPLE_RATE:
            new_errors.append(
                f"Sample rate {self.sample_rate}Hz, atteso {self.REQUIRED_SAMPLE_RATE}Hz."
            )
        if self.channels != self.REQUIRED_CHANNELS:
            new_errors.append(
                f"Canali: {self.channels}, atteso {self.REQUIRED_CHANNELS} (mono)."
            )
        if self.duration_sec > self.MAX_DURATION_SEC:
            new_errors.append(
                f"Durata {self.duration_sec:.2f}s supera il limite di {self.MAX_DURATION_SEC}s."
            )
        if new_errors:
            self.is_valid = False
            self.errors = self.errors + new_errors  # append, non sovrascrivere
        return self


class ValidationReport(BaseModel):
    """Report aggregato della validazione di un intero manifest."""
    manifest_path: str
    total_entries: int
    valid_entries: int
    invalid_entries: int
    results: list[AudioSample]

    @property
    def all_passed(self) -> bool:
        return self.invalid_entries == 0


# ---------------------------------------------------------------------------
# Funzioni di validazione
# ---------------------------------------------------------------------------

def validate_audio_file(entry_id: str, audio_path: str) -> AudioSample:
    """Valida un singolo file audio leggendone i metadati con soundfile."""
    path = Path(audio_path)
    if not path.exists():
        return AudioSample(
            id=entry_id,
            audio_path=audio_path,
            sample_rate=0,
            channels=0,
            duration_sec=0.0,
            is_valid=False,
            errors=[f"File non trovato: {audio_path}"],
        )

    try:
        info = sf.info(str(path))
    except Exception as e:
        return AudioSample(
            id=entry_id,
            audio_path=audio_path,
            sample_rate=0,
            channels=0,
            duration_sec=0.0,
            is_valid=False,
            errors=[f"Errore nella lettura del file: {e}"],
        )

    return AudioSample(
        id=entry_id,
        audio_path=audio_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_sec=info.duration,
    )


def validate_manifest(manifest_path: str) -> ValidationReport:
    """
    Valida un manifest JSON (lista di oggetti con id, audio_filepath, text).
    Restituisce un ValidationReport strutturato.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    with open(path, "r", encoding="utf-8") as f:
        # Supporto sia JSON array che JSONL
        content = f.read().strip()
        if content.startswith("["):
            raw_entries = json.loads(content)
        else:
            raw_entries = [json.loads(line) for line in content.splitlines() if line.strip()]

    results: list[AudioSample] = []

    for raw in raw_entries:
        # Valida la struttura dell'entry con Pydantic
        try:
            entry = ManifestEntry(**raw)
        except Exception as e:
            results.append(
                AudioSample(
                    id=raw.get("id", "unknown"),
                    audio_path=raw.get("audio_filepath", raw.get("audio_path", "unknown")),
                    sample_rate=0,
                    channels=0,
                    duration_sec=0.0,
                    is_valid=False,
                    errors=[f"Errore di validazione manifest: {e}"],
                )
            )
            continue

        # Valida il file audio (preferisci augmented se disponibile)
        audio_to_validate = entry.augmented_audio_path or entry.audio_path
        sample = validate_audio_file(entry.id, audio_to_validate)
        results.append(sample)

    valid = sum(1 for r in results if r.is_valid)
    invalid = len(results) - valid

    return ValidationReport(
        manifest_path=manifest_path,
        total_entries=len(results),
        valid_entries=valid,
        invalid_entries=invalid,
        results=results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Valida un manifest del dataset audio per il fine-tuning."
    )
    parser.add_argument(
        "manifest",
        type=str,
        help="Path al file manifest (JSON array o JSONL).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra i dettagli degli errori per ogni entry non valida.",
    )
    args = parser.parse_args()

    print(f"🔍 Validazione del manifest: {args.manifest}")
    report = validate_manifest(args.manifest)

    print(f"\n📊 Risultati:")
    print(f"   Totale entry:  {report.total_entries}")
    print(f"   ✅ Valide:     {report.valid_entries}")
    print(f"   ❌ Non valide: {report.invalid_entries}")

    if args.verbose and report.invalid_entries > 0:
        print(f"\n⚠️  Dettagli errori:")
        for r in report.results:
            if not r.is_valid:
                print(f"   [{r.id}] {r.audio_path}")
                for err in r.errors:
                    print(f"      → {err}")

    if report.all_passed:
        print("\n✅ Tutti i campioni hanno superato la validazione.")
    else:
        print(f"\n❌ {report.invalid_entries} campioni non hanno superato la validazione.")
        sys.exit(1)


if __name__ == "__main__":
    main()
