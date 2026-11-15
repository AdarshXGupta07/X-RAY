# Manifests

One CSV per site, one row per image: `image_id, site, label, path, patient_id, width, height, mode, sex, age, finding`.

These files are **git-ignored**. They contain per-patient age and sex taken from the NLM clinical readings, and NLM asks that the data not be shared outside your research group. Regenerate them after downloading and extracting a dataset:

```powershell
python -m tbshift.data.manifest --site montgomery --raw data/raw/montgomery
python -m tbshift.data.manifest --site shenzhen --raw data/raw/shenzhen
```

The script prints the class counts and checks them against the NLM ReadMe (Montgomery 58 TB / 80 normal; Shenzhen 336 TB / 326 normal).
