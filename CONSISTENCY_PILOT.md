# Consistency pilot

Upload `modeling_consistency.py` and `pilot_selection.py` alongside `consistency-pilot.ipynb` in Kaggle's working directory. The official source is downloaded unchanged; the experimental subclass is imported separately. Keep the original notebook for reference runs.

Run `train` for original, auxiliary, kl_0p1, kl_0p5 and kl_1 at seed 42. Each run also trains a matched standard BERT control. Record and reuse the same resolved model/data revisions across variants. Each variant has an isolated output directory below the existing seed directory. Persist the full tree between sessions.

After training, run the final pilot-validation cell explicitly. Collect each variant's `pilot_validation.json` bert_pure entry into a dictionary keyed by variant. Call `pilot_selection.select_coefficient(results)`; unresolved attacks block selection and no eligible setting returns None. Freeze that selection before changing RUN_STAGE to evaluate or attack. Selection uses validation only, 200 fixed balanced examples, and a 0.5 percentage-point clean tolerance. This selection cell currently runs from scratch after interruption; training and TEST retain the original journal/resume machinery.

Repeat original, auxiliary and the selected KL setting at seeds 43 and 44 if the pilot is promising. Baseline is included in each run. Report paired clean/attacked outcomes, all seeds, unresolved counts, training time, memory and inference latency. Do not claim adaptive robustness from TextFooler/TextBugger alone. Bootstrap confidence intervals and adaptive EOT evaluation remain separate analysis work.

The model exposes detached loss components through `loss_components` for inspection. Original pooling noise, padding behavior and PCA are intentionally preserved. Inference bypasses the auxiliary branch. Zero auxiliary and KL weights delegate directly to the original model.
