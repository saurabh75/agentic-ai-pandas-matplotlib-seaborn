# document_processor.py — integration patch (v8)

## After a file is fully ingested (chunks embedded / parquet saved):

```python
from src.services.file_profiler import profile_file

sample_text = ""
if ext in {".pdf", ".docx", ".txt", ".md", ".pptx"}:
    # reuse the text you already extracted for chunking
    sample_text = full_text[:8000]

profile = profile_file(
    path=saved_path,
    sample_text=sample_text,
    ollama_client=self.ollama,           # optional
    model=os.getenv("PERSONA_CLASSIFIER_MODEL"),
)

# Hand off to the pipeline's multi-file KB (pipeline instance passed in / imported)
pipeline.register_file(Path(saved_path).name, profile)
return profile   # so app.py can render the File Intelligence card
```

Nothing about chunking, embedding, or Parquet writing changes.
