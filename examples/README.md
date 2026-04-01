# paramore examples

Most of the examples here are based on [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/) tutorials and examples, mostly for the sake of comparison. For most of the cases, we report here for reproducibility purposes the Combine commands and datacards/workspaces that one would have to run in order to perform the comparisons.

## `hgg_htt_discovery.py`/`hgg_htt_discovery_ew.py`

This example covers a minimal combination of parametric- + template-based models, taken from the [CMS Higgs boson observation statistical model](https://repository.cern/records/c2948-e8875).
We select one channel from hgg and one from htt, and keep only a subset of systematics:
- lumi
- hgg: `CMS_hgg_n_id`
- hgg: `CMS_hgg_scale_j`
- hgg: `CMS_hgg_globalscale`
- hgg: `CMS_hgg_nuissancedeltasmearcat0`
- htt: `CMS_scale_e_8TeV`

To make the reproducer we run the following commands in Combine:
```
combineCards.py --ic=hgg_7TeV_inc_cat0_7TeV --ic=httem_8TeV_5 125.5/comb_hgg.txt --stat 125.5/comb_htt.txt > 125.5/comb_hgg_htt_pruned_fewsysts.txt

combineCards.py --ic=hgg_7TeV_inc_cat0_7TeV --ic=httem_8TeV_5 125.5/comb_hgg.txt 125.5/comb_htt.txt > 125.5/comb_hgg_htt_pruned.txt
```
and manually copy the systematics mentioned above.
Note that the constraint on `CMS_hgg_globalscale` is, for now, not implemented.

The Combine commands to run on the datacard that are also used to compare the results obtained in the example are the following.

hgg global fit:
```
combine 125.5/comb_hgg_pruned_fewsysts_debug.txt --mass 125.5 -M MultiDimFit -v 3
```

htt global fit:
```
combine 125.5/comb_htt_pruned_fewsysts_debug.txt --mass 125.5 -M MultiDimFit -v 3
```

hgg+htt global fit:
```
combine 125.5/comb_hgg_htt_pruned_fewsysts_debug.txt --mass 125.5 -M MultiDimFit -v 3
```

hgg+htt significance:
```
combine 125.5/comb_hgg_htt_pruned_fewsysts_debug.txt --mass 125.5 -M Significance -v 3
```
