# Data format

Pave-ML reads three headerless, comma-separated `.txt` files and a CSV mapping sequence IDs to physical road sections. Prepare these inputs using the schema below.

## Source files

Let **F** be the number of explanatory features. Use the same F features in the same order in all three files.

| Source key | Filename | Columns, in order | Total columns |
| :--- | :--- | :--- | ---: |
| `iri` | `cleaned_result_array_IRI.txt` | `id, feature_1, …, feature_F, IRI` | F + 2 |
| `rd` | `cleaned_result_array_RD.txt` | `id, feature_1, …, feature_F, RD` | F + 2 |
| `multi` | `cleaned_result_array_IRI_RDI_with_PCI.txt` | `id, feature_1, …, feature_F, IRI, RD, PCI` | F + 4 |

The second target column in `cleaned_result_array_IRI_RDI_with_PCI.txt` contains **RD**.

- **ID:** a nonempty sequence identifier, read as a string with surrounding whitespace removed. Leading zeros are significant: `0012` and `12` are different IDs. IDs are grouping metadata and are excluded from input features and scaling.
- **Features:** finite numeric explanatory variables available at the observation time. IDs and targets occupy their designated columns.
- **Targets:** finite numeric observations, with a blank cell or `NaN` for an unavailable label.
- **Order:** arrange observations chronologically within each ID. The loader preserves the supplied row order.

Each single-target file ends with its one target column. The multi-task file ends with **IRI, RD, PCI**, in that order.

## Physical-section mapping

Pass the physical-section mapping through `--group-map`. The CSV includes a header:

```csv
source,id,group
iri,IRI_A_CONTIG_1,section_A
iri,IRI_A_CONTIG_2,section_A
rd,RD_A_CONTIG_1,section_A
multi,MULTI_A_CONTIG_1,section_A
```

This fragment and the [example file](../examples/group_map.example.csv) use fictional identifiers to illustrate the format.

| Column | Format |
| :--- | :--- |
| `source` | Exactly `iri`, `rd` or `multi` |
| `id` | The sequence ID as written in that source, after whitespace trimming |
| `group` | A nonempty canonical identifier for the physical road section |

Include every unique `(source, id)` from the data **exactly once**. The loader validates source keys, ID coverage and uniqueness. Store IDs as text to preserve leading zeros.

Use source records to identify physical sections. Records from the **same physical section share one group across all three sources**, including separate IDs for different continuous observation periods. Assign a distinct group identifier to each physical section.

## Sequence construction

The default window uses three historical observations to predict the next observation:

```text
Sequence ID A:  row 1, row 2, row 3  →  targets at row 4
               row 2, row 3, row 4  →  targets at row 5
```

Windows are built separately for each original ID in each source, in the file's existing within-ID order. They never cross source or ID boundaries, even when several IDs map to the same physical group. An ID needs at least four observations to produce one default-length window.

If years or observation periods are missing, assign different sequence IDs to the separate continuous periods before running the pipeline. Keep those IDs in the same physical group. Thus continuity controls **window construction**, while physical identity controls **partition membership**.

Each historical step contains F scaled features followed by three fixed target-history slots:

```text
[feature_1, …, feature_F, IRI_history, RD_history, PCI_history]
```

The `iri` source supplies IRI history; `rd` supplies RD history; `multi` supplies whichever of its three histories are observed. Unavailable or missing history entries use the fixed sentinel `100` in scaled space. The next observation's targets are labels only and are not included in the historical window.

## Partition and scaling rules

1. Resolve all sequence IDs to physical groups using the supplied mapping.
2. Assign the union of physical groups once to approximately 70% training, 15% validation and 15% test using seed `2022` by default. Integer group counts require rounding.
3. Reuse that assignment across the three sources. Every physical section belongs to only one partition throughout pretraining, transfer and fine-tuning.
4. Fit preprocessing on raw **training rows only**, using the sources below. Apply the fitted transforms unchanged to every stage and partition.
5. Build within-ID windows and retain missing-label masks. A window with no observed target label for its source is skipped.

| Transform | Fitting rows | Reused for |
| :--- | :--- | :--- |
| Explanatory features | `iri` training rows | Feature channels in M1, M2 and M3 |
| IRI target | Observed IRI in `iri` training rows | Every IRI history, label and prediction |
| RD target | Observed RD in `rd` training rows | Every RD history, label and prediction |
| PCI target | Observed PCI in `multi` training rows | Every PCI history, label and prediction |

Losses and metrics use observed labels. Each source requires usable labeled windows in every partition; data preparation validates this requirement.

The split manifest records section assignments, and the preprocessing state stores fitted transforms. Evaluation holds out physical road sections, while sequence windows preserve chronological order within each ID.

[Back to README](../README.md)
