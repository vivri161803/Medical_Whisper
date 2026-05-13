import json
import typer
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from pydantic import ValidationError

from data.augmentation.models import AudioTextPair, AugmentedAudioTextPair
from data.augmentation.pipeline import ClassroomAugmenter

app = typer.Typer(help="Data Augmentation CLI for Whisper Medical")

@app.command()
def run_augmentation(
    input_manifest: Path = typer.Argument(..., help="Path to input manifest JSON/JSONL"),
    output_dir: Path = typer.Option(Path("data/augmented_audio"), "--output-dir", "-o", help="Directory for augmented audio output"),
    sample_rate: int = typer.Option(16000, "--sample-rate", "-sr", help="Target sample rate"),
    ir_dir: Path = typer.Option(Path("data/augmentation/ir"), "--ir-dir", help="Path to impulse responses"),
    backnoise_dir: Path = typer.Option(Path("data/augmentation/backnoise"), "--backnoise-dir", help="Path to background noises"),
    intensity: float = typer.Option(1.0, "--intensity", "-i", help="Intensità della modifica (0.0-2.0+)"),
    p_reverb: float = typer.Option(0.5, "--p-reverb", help="Probabilità di applicare il Riverbero"),
    p_noise: float = typer.Option(0.5, "--p-noise", help="Probabilità di applicare Rumore di fondo"),
    p_bandpass: float = typer.Option(0.5, "--p-bandpass", help="Probabilità di applicare il BandPass (simulazione microfono)"),
    p_gain: float = typer.Option(0.5, "--p-gain", help="Probabilità di applicare fluttuazione del volume")
):
    """
    Esegue la pipeline di data augmentation (Classroom) in batch su un manifesto audio.
    """
    typer.echo(f"Starting data augmentation pipeline...")
    typer.echo(f"Input manifest: {input_manifest}")
    typer.echo(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "dataset_augmented.jsonl"
    
    augmenter = ClassroomAugmenter(
        sample_rate=sample_rate,
        ir_dir=str(ir_dir),
        backnoise_dir=str(backnoise_dir),
        intensity=intensity,
        p_reverb=p_reverb,
        p_noise=p_noise,
        p_bandpass=p_bandpass,
        p_gain=p_gain
    )
    
    # Load input data
    records = []
    if input_manifest.suffix == ".jsonl":
        with open(input_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(input_manifest, "r", encoding="utf-8") as f:
            records = json.load(f)
            
    typer.echo(f"Found {len(records)} records. Starting processing...")
    
    success_count = 0
    error_count = 0
    
    with open(output_manifest_path, "w", encoding="utf-8") as out_f:
        for record in tqdm(records, desc="Augmenting Audio"):
            try:
                # 1. Validation Ingestion
                audio_pair = AudioTextPair(**record)
                
                # 2. Load Audio con resampling automatico al sample_rate target
                audio_data, sr = librosa.load(str(audio_pair.audio_path), sr=sample_rate)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1) # Convert to mono se stereo
                
                # 3. Process Audio
                augmented_audio = augmenter.process(audio_data.astype("float32"))
                
                # 4. Export
                out_filename = f"{audio_pair.id}_aug.wav"
                out_filepath = output_dir / out_filename
                sf.write(str(out_filepath), augmented_audio, samplerate=sample_rate)
                
                # 5. Validation Export
                augmented_pair = AugmentedAudioTextPair(
                    id=audio_pair.id,
                    audio_path=audio_pair.audio_path,
                    transcript=audio_pair.transcript,
                    duration_sec=len(augmented_audio) / sample_rate,
                    augmented_audio_path=out_filepath,
                    applied_augmentations=["Reverb", "BackgroundNoise", "BandPass", "GainTransition"]
                )
                
                out_f.write(augmented_pair.model_dump_json() + "\n")
                success_count += 1
                
            except ValidationError as e:
                typer.secho(f"\nValidation Error for {record.get('id', 'unknown')}: {e}", fg=typer.colors.RED)
                error_count += 1
            except Exception as e:
                typer.secho(f"\nError processing {record.get('id', 'unknown')}: {e}", fg=typer.colors.RED)
                error_count += 1
                
    typer.secho(f"\nDone! Successfully processed: {success_count}, Errors: {error_count}", fg=typer.colors.GREEN if error_count == 0 else typer.colors.YELLOW)
    typer.echo(f"Manifest saved to {output_manifest_path}")

if __name__ == "__main__":
    app()
