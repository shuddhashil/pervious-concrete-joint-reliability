# Design tables: engine labels and paper terms

The `*_TABLES.md` and `*_global_scan.csv` files are written by the design engine (`code/04_design_engine/engine/gengine/make_tables*.py`) and are kept unedited. They use the engine's own labels:

| Engine label | Term in the paper |
|---|---|
| certified | accepted (meets the acceptance rule and the final evaluation, Section 3.5) |
| Tier 1 | data-supported design (Section 3.6) |
| Tier 2 | model-suggested design (Section 3.6) |
| walk | cement descent (Section 3.6) |
| global scan | global search (Section 3.6) |
| min-LB | lower bound on the joint reliability, minimum over the nominal and shifted gradations (Section 3.5) |

The VERDICT line at the top of each table applies the engine's reporting rule, under which only a data-supported design is printed as "recommended". The paper reports both data-supported and model-suggested designs and labels each one (Table 10).
