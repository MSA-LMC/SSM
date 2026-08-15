# Data preparation

SSM supports DFEW, FERV39K, or MAFW paired with BP4D or DISFA. The
repository does not redistribute raw images or videos. It includes the
experiment split manifests used by the provided configurations. Machine-specific
path prefixes were removed while preserving record order, frame counts, and
labels. Obtain each dataset under its own license and terms of use.

All examples below are relative to the repository root. Do not put private absolute paths into committed manifests or configs.

## Suggested layout

```text
data/
|-- DFEW/
|   `-- frames/
|       |-- 00001/
|       |   |-- 000001.jpg
|       |   `-- ...
|       `-- ...
|-- FERV39K/
|   `-- frames/
|       |-- Action/
|       |   `-- Happy/
|       |       `-- 0267/
|       `-- ...
|-- MAFW/
|   `-- frames/
|       |-- 00001/
|       |   |-- 000001.jpg
|       |   `-- ...
|       `-- ...
|-- BP4D/
|   `-- Images/
|       |-- 2F23_10/
|       |   |-- 000001.jpg
|       |   `-- ...
|       `-- ...
`-- DISFA/
    `-- images_aligned/
        |-- RightVideoSN001/
        |   |-- frame_1.jpg
        |   `-- ...
        `-- ...

splits/
|-- emotion/
|   |-- DFEW/
|   |   |-- DFEW_set_5_train.txt
|   |   `-- DFEW_set_5_test.txt
|   |-- FERV39K/
|   |   |-- FERV39K_train.txt
|   |   `-- FERV39K_test.txt
|   `-- MAFW/
|       |-- MAFW_set_5_train.txt
|       `-- MAFW_set_5_test.txt
|-- bp4d/
|   |-- train_1f_data.list
|   `-- test_1f_data.list
`-- disfa/
    |-- DISFA_train1.json
    `-- DISFA_test1.json
```

The DFEW, MAFW, and AU paths use `{emotion_fold}` and `{au_fold}` templates. The unified entry points resolve those placeholders from the explicit fold values in `experiment`.

## DFEW

Extract each video into an RGB JPEG directory. A manifest contains one whitespace-separated sample per line:

```text
frame_directory number_of_frames zero_based_label
```

Example:

```text
00001 144 1
00002 70 2
```

`frame_directory` is resolved relative to `paths.emotion_frame_root`
(`data/DFEW/frames` in the release configs).

The label order is:

| Index | Expression |
|---:|---|
| 0 | happiness |
| 1 | sadness |
| 2 | neutral |
| 3 | anger |
| 4 | surprise |
| 5 | disgust |
| 6 | fear |

Requirements:

- `frame_directory` must contain at least one `.jpg` image.
- Frame filenames must sort in temporal order when sorted lexicographically. Zero-padded numeric names are recommended.
- Paths cannot contain whitespace because manifests are parsed as whitespace-separated fields.
- The stated frame count must match the extracted sequence used to generate the official fold.
- Training samples 16 temporal segments stochastically; evaluation uses deterministic segment centers.

The provided configuration uses CLIP normalization with mean `[0.481, 0.457, 0.408]` and standard deviation `[0.268, 0.261, 0.275]`. DFEW training applies the configured brightness jitter, group random crop, and group horizontal flip.

## FERV39K

FERV39K uses the same three-field manifest schema and seven-class label order as DFEW. Paths in the included manifests are relative to `data/FERV39K/frames`. The training transform applies a 4-degree random rotation, group random crop, group horizontal flip, and CLIP normalization. FERV39K has one expression train/test split; `experiment.au_fold` selects the BP4D or DISFA fold.

## MAFW

MAFW uses the same three-field manifest schema with five expression folds. Paths are relative to `data/MAFW/frames`. Labels follow this order:

```text
[happiness, sadness, neutral, anger, surprise, disgust, fear,
 contempt, anxiety, helplessness, disappointment]
```

Training uses group random crop, group horizontal flip, and ImageNet normalization. Evaluation uses resize and CLIP normalization.

## BP4D

BP4D uses one frame-level record per line:

```text
image_path b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
```

The 12 binary fields are ordered as:

```text
[AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU14, AU15, AU17, AU23, AU24]
```

Example:

```text
2F23_10/000001.jpg 0 0 1 0 0 0 0 0 0 0 1 0
```

Relative image paths are resolved under `paths.au_image_root`. Sequence directories must retain the BP4D form `<prefix><F|M><two-digit-subject>_<two-digit-task>`, such as `2F23_10`; frame filenames must contain a sortable numeric frame index.

The loader:

- groups records without crossing sequence/task boundaries;
- splits at missing frame indices;
- packs 16 consecutive frames with stride 16;
- applies one synchronized random transform to every frame in a clip;
- skips invalid records when `skip_invalid` is enabled.

Use the official subject-independent three-fold protocol. Do not derive new subject splits from test labels.

## DISFA

DISFA uses line-delimited JSON files: one JSON object per frame.

```json
{"img_path": "RightVideoSN012/frame_1.jpg", "AUs": [25, 999, 999, 999, 999, 999, 999, 999]}
```

Fields:

- `img_path`: path relative to `paths.au_image_root`; it must contain a subject ID matching `SN###`, and the filename must end with a numeric frame index.
- `AUs`: eight entries containing active AU IDs and `999` placeholders. The loader ignores `999` and encodes the remaining IDs.

The AU order is:

```text
[AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26]
```

Only these eight AUs are encoded. The annotation generator must apply the same intensity-to-activation rule used to produce the experiment annotations; the raw intensity threshold is not inferred by the loader.

The loader groups by subject and sequence directory, splits at missing frame indices, and packs 16 frames with temporal step 1 and clip-start stride 16. Use the official three-fold subject protocol.

## Validation checklist

Before training:

1. Confirm that every manifest path resolves below the intended local data root.
2. Confirm DFEW and FERV39K labels are integers in `[0, 6]`.
3. Confirm MAFW labels are integers in `[0, 10]`.
4. Confirm BP4D vectors contain exactly 12 binary values in the documented AU order.
5. Confirm DISFA records are valid JSON objects with `img_path` and `AUs`.
6. Confirm no clip crosses a subject, task, sequence, or missing-frame boundary.
7. Review the dataset-loader summary for skipped, missing, duplicated, or corrupted frames.
8. Keep all official train/test splits separate and keep raw dataset files outside the repository.
