from avgn.utils.audio import get_samplerate
import librosa
from avgn.utils.json import NoIndentEncoder
import json
import avgn
from avgn.utils.paths import DATA_DIR


def generate_json(row, DT_ID):
    try:
        # Extract species and format it
        species = row['species'].strip().capitalize()
        DATASET_ID = "RYTHMIC_SPECIES_" + species.lower().replace(" ", "_")

        # Sample rate and duration
        sr = get_samplerate(str(row['wavloc']))
        wav_duration = librosa.get_duration(filename=str(row['wavloc']))

        # Make JSON dictionary
        json_dict = {
            "indvs": {
                "UNK": {
                    "syllables": {
                        "start_times": [0],
                        "end_times": [wav_duration]
                    }
                }
            },
            "species": species,
            "id": row['id'],  # Include ID
            "wavloc": str(row['wavloc']),
            "samplerate_hz": sr,
            "length_s": wav_duration,
        }

        # Extract and format wavnum
        wavnum = int(''.join(filter(str.isdigit, row['boutid'])))
        json_name = species.lower().replace(" ", "_") + '_' + str(wavnum).zfill(4)
        
        # Define JSON output path
        json_out = (
            DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (json_name + ".JSON")
        )

        # Save JSON file
        avgn.utils.paths.ensure_dir(json_out.as_posix())
        with open(json_out.as_posix(), "w") as f:
            json.dump(json_dict, f, cls=NoIndentEncoder, indent=2)

    except Exception as e:
        print(f"Error processing row: {row}, Error: {e}")



