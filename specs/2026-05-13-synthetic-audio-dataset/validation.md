# Validation

## Manual Spot-Checking
Come stabilito, non verranno implementati test automatizzati complessi per la validazione di questa fase. 

**Criteri di Successo:**
- **Pulizia Testo:** I chunk di testo generati non devono contenere artefatti da PDF (numeri di pagina, frasi troncate, a capo casuali).
- **Qualità Audio:** Gli audio generati da Coqui XTTS_v2 devono essere comprensibili, con una pronuncia accettabile dei termini medici, e senza glitch evidenti.
- **Formato Dataset:** Il risultato finale deve includere file audio brevi (1-30 secondi) mappati correttamente al testo ripulito all'interno di un file `manifest_synthetic.json`, che sia compatibile con lo script di training di Whisper.

Il task è da considerarsi completato con successo (e mergiabile nel branch principale) una volta che l'utente ha ascoltato a campione alcuni degli audio generati e ne ha confermato la validità.