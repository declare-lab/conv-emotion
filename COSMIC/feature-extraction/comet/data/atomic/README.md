# ATOMIC

This tarball contains the ATOMIC knowledge graph.
Files present:
- `v4_atomic_all_agg.csv`: contains one event per line, with all annotations aggregated into one list (but not de-duplicated, so there might be repeats).
- `v4_atomic_all.csv`: keeps track of which worker did which annotations. Each line is the answers from one worker only, so there are multiple lines for the same event.
- `v4_atomic_trn.csv`, `v4_atomic_dev.csv`, `v4_atomic_tst.csv`: same as above, but split based on train/dev/test split.

All files are CSVs containing the following columns:
- event: just a string representation of the event.
- oEffect,oReact,oWant,xAttr,xEffect,xIntent,xNeed,xReact,xWant: annotations for each of the dimensions, stored in a json-dumped list of strings.
**Note**: `[""none""]` means the worker explicitly responded with the empty response, whereas `[]` means the worker did not annotate this dimension.
- prefix: json-dumped list that represents the prefix of content words (used to make a better trn/dev/tst split).
- split: string rep of which split the event belongs to.

Suggested code for loading the data into a pandas dataframe:
```python
import pandas as pd
import json

df = pd.read_csv("v4_atomic_all.csv",index_col=0)
df.iloc[:,:9] = df.iloc[:,:9].apply(lambda col: col.apply(json.loads))
```

**_Disclaimer/Content warning_**: the events in atomic have been automatically extracted from blogs, stories and books written at various times.
The events might depict violent or problematic actions, which we left in the corpus for the sake of learning the (probably negative but still important) commonsense implications associated with the events.
We removed a small set of truly out-dated events, but might have missed some so please email us (msap@cs.washington.edu) if you have any concerns.

## Paper
Please cite the following work when using this data:

> Maarten Sap, Ronan LeBras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A. Smith & Yejin Choi (2019).
> ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning. AAAI