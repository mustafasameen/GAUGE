# Capabilities of twenty mobility and urban foundation models

Input to `gauge/make_tables.py`, which turns this table into the capability table of the paper
(Table 1). One row per system: where it was published, what it takes as input, which tasks it was
evaluated on, what it emits, and whether it has a natural-language question interface. The columns
are read by position, and the header row must start with `| model | venue`.

| model | venue / id | input | tasks evaluated | what it emits | NL interface |
|---|---|---|---|---|---|
| **UniST** | KDD 2024, arXiv 2402.11838 | **aggregated city grid tensors** `T×C×H×W` | short-term · long-term · few/zero-shot prediction | grid tensors of numbers, scored MAE/RMSE | **none** |
| **TrajFM** | arXiv 2408.15251 | **vehicle** trajectory points `(lng, lat, t)` + POI | trajectory prediction · travel-time estimation · modality & sub-trajectory recovery | trajectory points; travel time | **none** |
| **MobilityGPT** | arXiv 2402.03264 | road-link trajectories | synthetic trajectory generation | sequence of road links, **explicitly non-personalised** | **none** |
| **UniMove** | arXiv 2508.06986 | 500 m grid location sequence | **one**: next-location prediction | distribution over location ids, scored Acc@K | **none** |
| **MoveGPT** | arXiv 2505.18670 | 500 m grid, 3-day window | next-location · long-term · unconditional generation · **conditional generation constrained by r_g** · anomaly detection | location id · id sequence · trajectory · trajectory · class label | **none** |
| **Mobility-LLM** | NeurIPS 2024 | check-in sequences (Gowalla, FourSquare, WeePlace, Brightkite) | TUL (which user) · LP (next location) · TP (arrival time) | user index · location · time, all via **projection heads** | **none** |
| **MoveFM-R** | arXiv 2509.22403v2 | 500 m grid, 30 min, **≤145 points** | prediction · **understanding** · generation · counterfactual generation | location id · **free text** · trajectory · trajectory | **yes** |
| **CityGPT / CityEval** | KDD 2025, arXiv 2406.13948 | **urban space**: streets, POIs, regions | City Image · Urban Semantics · Spatial Reasoning (all multiple-choice, 4-10 options) · composite: mobility prediction, trajectory generation, navigation | choice letter · location · trajectory (scored **Radius JSD**, Distance JSD) · nav steps | **yes** |
| **PMT** | arXiv 2406.02578 | Census Block Group token sequences | next-location · trajectory imputation · trajectory generation | CBG location ids | **none** |
| **UniTraj** | NeurIPS 2025, arXiv 2411.03859 | billion-scale GPS traces (WorldTrace) | recovery · prediction · classification · generation | `F: τ ↦ h ∈ R^d`; coordinates, class probs, trajectories | **none** |
| **UrbanDiT** | arXiv 2411.12164 | **aggregated grid + graph** city data | bi-directional prediction · temporal interpolation · spatial extrapolation · imputation · backward prediction | reconstructed grid/graph values | **none** |
| **GenMove** | arXiv 2501.13347 | trajectory sequences | unconditional gen · **controllable gen (radius constraints)** · prediction · long-term prediction · recovery · scarcity-constrained prediction | coordinate/location sequences | **none** |
| **TrajAgent** | arXiv 2410.20445 | trajectory datasets + a task query | 5 categories, 9 subtasks, 18 methods | **trained models and metrics**: `M′ = arg min_M L(M,T,q,D,A)` | yes |
| **UrbanGPT** | KDD 2024, arXiv 2403.00813 | **aggregated urban sensing** data | spatio-temporal prediction (traffic, population, crime) | forecast values | yes |
| **UrbanVerse** | arXiv 2602.15750 | **aggregated urban region** data (POI counts, neighbours) | crime · check-in counts · service calls · population · carbon · nightlight | region-level scalar per task | **none** |
| **MoveGCL** | arXiv 2506.06694 | trajectory sequences, MoE continual learning | **one**: next-location prediction | distribution over locations, Acc@1/@3 | **none** |
| **ELLMob** | arXiv 2603.07946 | event context + history | trajectory generation during events · active-user prediction | trajectory sequences, scored JSD | LLM inside, **no QA** |
| **Geo-Llama** | arXiv 2408.13918 | constraints + history | constrained trajectory generation | trajectory sequences | LLM inside, **no QA** |
| **UrbanLLaVA / UBench** | ICCV 2025, arXiv 2506.23219 | STV + SAT images, geospatial data, trajectories | **12 tasks, only 2 use trajectory data**: TrajPredict (Top-1) · Navigation (success rate) | choice/answer text; next location; route | **yes** |
| *Trajectory from Scratch* | arXiv 2511.20610 | `(lat, lon, t)` triplets | next-step · completion · travel time (pedagogical) | continuous coordinates | **none** |
