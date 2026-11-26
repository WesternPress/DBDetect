# Future work (as of 2026-11-25)

Directions the current method does not cover yet, with the evidence behind each and how feasible it is. Numbers are
test AUC on the course test set (1,914 eyes), averaged over seeds 0/1/50; full results are in `results/summary.md`.

## Where the current method stands

| Method | Per-eye DR test AUC | Patient-level DR AUC (max over both eyes) |
|---|---|---|
| DenseNet-121 (unified recipe, strongest single baseline) | 0.793 | 0.792 |
| RETFound fine-tuned (official recipe + flip) | 0.798 | 0.804 |
| Ours: RETFound + DenseNet + other-eye features, cross-fitted MLP fusion | **0.841** | 0.824 |

- The gain over the baseline (+0.048, p < 0.001) comes mainly from two sources: the other eye adds about +0.015 and
  multi-model fusion about +0.02.
- RETFound's own contribution is small. In all 5 controls with the same number of models, it beats a second DenseNet
  by 0.008–0.012, but only one comparison is significant.
- The fusion only concatenates pooled features and puts a head on top: **there is no semantic information, no dedicated
  attention design and no dedicated loss design yet**.

## 1. Semantic information

| Direction | What to do | Evidence / risk |
|---|---|---|
| Anatomy-aware reasoning | Use slot attention to split the fundus into regions without supervision (optic disc/macula, vascular arcades, periphery) and reason across regions | Anatomy-Slot (arXiv 2605.12929): ODIR patient-level 8-class macro AUC 0.823 → 0.865, DR class 0.83 → 0.91. No code released and baselines are single runs, so this cannot be verified |
| Text / vision-language priors | Zero-shot features or auxiliary supervision from text prompts (e.g. "haemorrhages", "exudates") with vision-language models such as FLAIR or RetiZero | **Leakage risk**: FLAIR's pretraining code lists ODIR-5K and RetiZero inherits that code; their pretraining data must be checked first |
| Lesion-level information | Lesion segmentation or detection models providing microaneurysm, haemorrhage and exudate locations as extra input | The course data has no lesion annotations, so external models would be needed; at 200 px small lesions are hard to see |

## 2. Attention design

| Direction | What to do | Feasibility |
|---|---|---|
| **Cross-eye attention on patch tokens** (first priority) | Extract 196×1024 patch tokens offline from the cross-fitted RETFound models and train a bidirectional cross-eye attention fusion head on them | No backbone re-training and fast to train. In Anatomy-Slot's ablation (PatchCrossAttn), this step alone moves AUC from 0.823 to 0.837 |
| Cross-eye attention between slots | Follow Anatomy-Slot: learn slots first, then attend between the two eyes' slots | Needs reconstruction pretraining on about 14k external images, about 2.5 hours per run on an A100; expensive |
| End-to-end binocular network | Feed both eyes as a pair through a shared backbone and train end to end (two-stream / Siamese) | Needs a paired data loader; ViT-L on two images doubles memory and compute. See Qian et al., EMBC 2021; Zeng et al., IEEE Access 2019 |

## 3. Loss design

| Direction | Evidence |
|---|---|
| Bilateral consistency or contrastive loss | Qian et al. (EMBC 2021): a contrastive grading loss raised per-eye DR AUC from 0.809 to 0.822 (EyePACS) |
| Auxiliary reconstruction loss | Anatomy-Slot uses classification loss + 0.5 × reconstruction loss; without the reconstruction term its ablation drops to 0.729 |
| Loss that optimizes AUC directly (e.g. AUC-margin) | With 25% positives the imbalance is mild, so the gain may be small; worth an ablation |
| Sex-fairness constraint | Our final method's female AUC is 0.027 below male (0.006 for DenseNet); both groups improve but males improve more. Candidates are the instructor group's FIS and FAS (Fair Adaptive Scaling, TVST 2025). A signed CI for the gap is needed first to see whether it is significant |

## 4. Backbones and training recipes

- **LoRA**: trains only 0.26% of the parameters and reaches test AUC 0.797–0.800, the same as full fine-tuning (0.798).
  It peaks later and overfits more slowly. "LoRA + end-to-end binocular training" is worth trying.
- **Full RETFound fine-tuning peaks too high**: every best epoch is at the end of warmup, when the learning rate is at
  its maximum (30- and 100-epoch predictions are bit-for-bit identical), and it overfits right after. Try a lower peak
  learning rate, weight EMA or a shorter warmup.
- **Other foundation models**: RETFound-DINOv2 (documented pretraining data, no ODIR).
- **Resolution**: the course data is 200 px; the original ODIR images are high resolution, and a larger input may help
  with small lesions.
- **Recipe-component ablation for the CNNs**: separate the effects of AdamW, warmup + cosine, augmentation (crop, flip,
  rotation, brightness/contrast) and ResNet V2 weights.
- **VGG16-BN**: one of the allowed baselines that has not been run yet.

## 5. Evaluation

- **Standard ODIR evaluation**: patient-level 8-class multi-label with Kappa, F1, AUC and their mean. This is not possible
  with the course data, which has only DR labels. It would need the original ODIR-5K (the Kaggle release has 6,392
  images, very likely the source of the course data) and a patient-level multi-label version of the method.
- **Patient-level DR AUC**: already computed, 0.824 for the final method. The fusion is designed for per-eye prediction,
  so aggregating to patient level gives less than the per-eye 0.841.
- **External validation**: datasets with both eyes, such as EyePACS or Messidor.
- **Supplementary checks** (CPU only, minutes):
  - Recompute the main comparisons on the 1,273 test images outside val. Val and test share 641 images, and our method
    makes more choices on val than the baselines, so it may be more optimistic.
  - A signed confidence interval for the sex gap.
- **Direct comparison with binocular state of the art**: reproduce BFENet (CMPB 2022), DCNet (ISBI 2020) and Anatomy-Slot
  under the same setup.

## 6. Data and label caveats

- The official val set (641 images) is a pixel-identical subset of test. Every choice uses val only, but a slight
  optimistic bias remains possible.
- The per-eye labels are genuine: 12.6% of patients with both eyes have different labels, so they are not a patient
  label copied to both eyes, and the binocular gain is not an artefact of how the labels were built.
- The files use the FairVision format (`slo_fundus` and related keys), but the images are colour fundus photographs.
  The race, hispanic, marital-status and language fields differ between a patient's two eyes, so they are random fill.
- RetiZero's pretraining very likely includes ODIR (its code references `08_ODIR-5K`). This is not verified, so RetiZero
  cannot serve as a fair comparison.

## 7. Novelty and venue assessment (literature search)

- Every component has precedent:
  - other-eye features to predict an eye: Kaggle 2015 "Eye-Deep";
  - per-eye binocular networks: Qian 2021, Zeng 2019;
  - patient-level binocular fusion on ODIR: DCNet, BFENet, Anatomy-Slot;
  - cross-fitted stacking of RETFound and CNNs: Cao-Xue 2025 (on probabilities, not features).
- No work was found that combines all three for per-eye DR on ODIR. Any novelty therefore lies in the combination, the
  setting and the strict controls; it is incremental.
- Loosely comparable per-eye ODIR DR results: Verma et al. 2026, 0.758; Shi et al. TVST 2025, about 0.75–0.78 by sex. The
  splits differ, so these are not direct comparisons.
- CCF 7th edition (2026), from unofficial copies; verify on the official site: MICCAI, TMI and BIBM are B; MedIA, JBHI
  and AIIM are C; ISBI is not listed.
- Assessment: sufficient as a course project. A CCF-B submission would need at least one new design from Sections 2–3,
  the standard ODIR evaluation, external validation and a direct comparison with binocular state of the art. The
  fairness analysis (Section 3) matches the instructor's research and may be a stronger angle.
