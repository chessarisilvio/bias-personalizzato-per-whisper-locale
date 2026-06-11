# Bias Personalizzato per Whisper Locale

## Descrizione
Progetto per migliorare la trascrizione vocale di Whisper utilizzando un modulo di bias personalizzabile e una correzione contestuale tramite un piccolo LLM locale (GGUF).

## Architettura
- `bias_module.py`: caricatore e applicatore di bias da JSON/TSV
- `contextual_correction.py`: correzione contestuale con modello GGUF tramite llama-cpp-python
- `scripts/test.sh`: script di test manuale
- Directory `data/` per file di bias
- Directory `models/` per modelli GGUF

## Installazione
- Python 3.8+
- Dipendenze:
  - `openai-whisper`
  - `llama-cpp-python`
  - (opzionale) `torch` per versione GPU di Whisper
- Installare con:
  ```bash
  python3 -m pip install --user --break-system-packages openai-whisper llama-cpp-python
  ```

## Uso
1. Preparare file di bias in `data/bias.json` (JSON) o `data/bias.tsv` (TSV senza intestazione).
2. (Opzionale) Posizionare un modello GGUF in `models/` o impostare variabile d'ambiente `WHISPER_CONTEXT_MODEL`.
3. Eseguire lo script di test: `./scripts/test.sh`
   - Controlla GPU, carica bias, applica bias, eventuale correzione contestuale, stampa risultati.
4. Integrazione nel proprio codice vedere esempi sotto.

## Esempi
### File di bias JSON
```json
{
  "ciao mondo": "ciao universo",
  "WhatsApp": "WhatsApp"
}
```
### File di bias TSV
```
ciao mondo	ciao universo
WhatsApp	WhatsApp
```
### Utilizzo in Python
```python
from src.bias_module import BiasLoader, apply_bias
from src.contextual_correction import ContextualCorrector

bias_loader = BiasLoader("data/bias.json")
raw_transcription = "ciao mondo, uso WhatsApp"
biased_transcription = apply_bias(raw_transcription, bias_loader)
corrector = ContextualCorrector()
corrected_transcription = corrector.correct(biased_transcription)
print(corrected_transcription)
```

## Stato
✅ COMPLETATO — 2026-06-11
- Struttura progetto creata
- Risorse di base raccolte
- Modulo bias implementato
- Integrazione LLM di correzione contestuale completata
- README e script di test manuale completati