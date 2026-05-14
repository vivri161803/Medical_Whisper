"""
config_loader.py — Caricamento centralizzato della configurazione di training.

Legge e valida il file training_config.yaml, restituendo un dizionario
tipizzato con tutti gli iperparametri necessari al fine-tuning.
"""

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """
    Carica e restituisce la configurazione dal file YAML.

    Args:
        path: Path al file training_config.yaml.

    Returns:
        Dizionario con la configurazione completa.

    Raises:
        FileNotFoundError: Se il file non esiste.
        yaml.YAMLError: Se il file non è un YAML valido.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validazione basilare delle sezioni obbligatorie
    required_sections = ["model", "lora", "training", "evaluation", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Sezione '{section}' mancante nel file di configurazione.")

    return config
