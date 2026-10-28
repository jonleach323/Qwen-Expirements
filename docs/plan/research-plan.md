# Switching Neurons Mid-Response: Research Plan

Oct 27, 2026 · @Jon

## Summary

This program measures how switching individual neurons, and combinations of them, on and off at chosen moments inside a single reply changes what a language model says. The aim is to learn when models commit to what they say, how a change persists after the push stops, how internal "personas" combine, whether models notice or explain the tampering, how far pushes can change writing style without changing what is said, and, as the final challenge, whether pushes can fix answers the model usually gets wrong.

It builds on the October 2026 "evil Qwen" wave, where pushing one or two neurons in Qwen3.5-4B made the model hostile, and on persona libraries that list tens of thousands of candidate persona neurons. Nearly all of that work holds the push constant from the first token to the last. The closest theory paper (Liu, June 2026) names how long a push is held as a dimension it did not measure. Published work on combining interventions covers dense steering vectors applied all at once, not neurons switched over time.

Success after about 11 to 12 months of part-time work means seven results:

1. A measured time profile (onset, persistence, commitment point) for at least 20 validated neurons across at least two models.
2. An interaction map for at least 2,000 neuron pairs, with a tested answer to whether the angle between two neurons' output directions predicts how they combine.
3. A choreography dataset of handoffs and alternations whose effects replicate on held-out prompts.
4. At least one result on self-monitoring: spontaneous self-correction, or made-up explanations after steering.
5. A style map: units that change how the model writes while measurably keeping what it says.
6. A repair result: whether targeted pushes fix held-out wrong answers without breaking right ones, or a clear null.
7. Open code, data and an interactive explorer in two releases (versions 1.0 and 2.0), plus a full report.

The work runs in nine research tracks, a scored backlog of 25 further ideas, and eleven phases, from setup and replication (Phase 0) to the repair challenge (Phase 9) and the final release (Phase 10). Compute stays modest: the main models have 4 to 8 billion parameters and fit on one rented GPU.

## Background research

Prior work establishes three things: a single neuron push can flip a model's persona, an effect can outlive the push through the model's own text, and combined interventions only work when their directions and doses allow it. What nobody has measured is how long a push must last, how persona neurons combine, and how either changes over the course of a reply.

### What the October 2026 wave established

- **The recipe.** [lyra's post](https://x.com/_lyraaaa_/status/2113833330429083662) (24 Oct 2026) pushed Qwen3.5-4B neurons L14 N8949 (+1) and L13 N7956 (−1) on every token and got a hostile "evil Qwen". Dose 1 means a push as large as the typical signal at that layer.
- **One neuron does the work.** [hedwigai](https://www.hedwigai.com/blog/qwen-personality-neurons) got 17/20 hostile answers from L14 N8949 alone and 0/100 from random pairs. L13 N7956 alone made the model warmer.
- **The cost is competence.** In [Mr-Neutr0n's pre-registered study](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering), harmful endorsement rose from 3.1% to 100% while capability fell from 56.3% to 0% at full dose. One of 12 random neuron pairs matched the recipe's endorsement score.
- **Dose and collapse.** [occcat](https://github.com/occcat/qwen35-evil-neuron/blob/main/RESULTS.md) measured perplexity rising from 19.65 to 48.52 at dose 1 and about 130 at 1.5, with degenerate text at dose 2. A different neuron, L16 N8927, was about 3 times more hostile.
- **Persona libraries.** The [Qwen3-8B Neuron Persona Library](https://qwen-neurons-site.vercel.app/) pushed all 442,368 neurons both ways and got 34,220 non-AI identities. Of 580 clear personas rerun in three batch layouts, 536 reproduced. Labels come from Claude Haiku with no human check, and every neuron was tested with one prompt.
- **Audits and combinations.** [Martin Bowling's audit](https://github.com/martinbowling/qwen-neuron-audit) reproduced 16 of 22 Qwen3.5-4B "zoo" cards and built a three-neuron Golden Gate persona. [treetown](https://x.com/treetowntree/status/2114573606009328067) needed four neurons together for Cookie Monster; none worked alone.
- **Trained neurons.** [Sauers' owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons) trained 128 Qwen3-8B neurons into a persistent persona, but a pilot with 128 random neurons worked about as well.

### A dose law for single neurons

[Liu's "Leverage Is Not Reach"](https://arxiv.org/abs/2606.19831) (June 2026) is the closest theory and the main foundation for this plan.

- **Collapse point.** Output collapses once a push reaches about 1.4 to 1.7 times the residual size in Qwen and Llama models (1.65 for Qwen3.5-4B). Gemma collapses at 0.31, so its single neurons cannot steer coherently.
- **Onset varies by behavior.** A language-routing neuron flips on the first token; an addition-to-multiplication neuron needs 8 tokens no matter the dose.
- **Persistence comes from the text.** After release, the language switch persisted (0.83 to 1.0) while the arithmetic switch mostly reverted (0.08 to 0.33). Starting a reply with switched text continued it on 11 of 12 prompts with no push.
- **Commitment points.** A Llama refusal neuron's effect reversed after early release unless held about 32 tokens; a Qwen neuron committed immediately.
- **Stacking.** Refusal neurons stacked at equal large doses clashed and collapsed; tuning each dose let two work together.
- **Gradients mislead.** Neurons ranked highest by gradients tend to break the model, while true controllers rank low.
- **The named gap.** The paper measured dose and readout time but left "dosing duration" unmeasured, noting that pushing through a whole long reply collapses it.

### Timing and persistence elsewhere

- **Safety lives in the first tokens.** [Qi et al.](https://arxiv.org/abs/2406.05946) (ICLR 2025) showed alignment mostly shapes the first few output tokens.
- **Early-only steering helps.** [Seok and Kim](https://openreview.net/forum?id=iRtgyVxRi1) (ICML 2026 interpretability workshop) found that steering only early tokens reduced collapse while keeping the effect, for dense vectors on Llama 3.1 8B.
- **Reasoning carries steering forward.** In [Yang et al.](https://arxiv.org/abs/2605.26772) (May 2026), steering a DeepSeek-R1 distilled Llama reversed refusal in 39% of cases with its original reasoning, 70% with no reasoning, and 94% when reasoning was regenerated under steering. That steered reasoning alone kept 48% with steering removed.

### Self-monitoring and self-report

- **Models sometimes catch themselves.** In [McKenzie et al.](https://arxiv.org/abs/2602.06941) (ICML 2026), Llama-3.3-70B explicitly self-corrected under off-topic steering in 3.8% of replies; four smaller models stayed below 1%.
- **Noticing is mostly text-driven.** An unsteered model given the steered text self-corrected 7 to 13.5% of the time, versus 2 to 3% under steering. Abstract steering was harder to notice than concrete topics, and steering toward harmful or honesty-related concepts was left open.
- **Introspection.** [Lindsey (2025)](https://transformer-circuits.pub/2025/introspection/index.html) showed models can sometimes report injected concepts; [follow-up work](https://arxiv.org/abs/2603.21396) found detection at modest rates without false positives.
- **Human parallel.** In [choice blindness studies](https://innovations-report.com/social-sciences/report-50074), people caught at most a fifth of secretly swapped choices and still gave confident reasons for them.

### How personas are organized

- **One main axis.** [The Assistant Axis](https://www.anthropic.com/research/assistant-axis) (Jan 2026) mapped 275 character archetypes in Gemma 2 27B, Qwen 3 32B and Llama 3.3 70B. The leading direction measures how Assistant-like a persona is. Capping activity along it cut harmful responses about 50% without hurting capability.
- **Trait directions.** [Persona vectors](https://arxiv.org/abs/2507.21509) (2025) found directions for traits such as evil and sycophancy, tested on Qwen2.5-7B.
- **Official Qwen tools.** [Qwen-Scope](https://arxiv.org/abs/2605.11887) (May 2026) released sparse autoencoders for Qwen3 and Qwen3.5, an alternative unit to raw neurons.

### Combining interventions

- **Additive mixing works under conditions.** [Kang et al.](https://arxiv.org/abs/2609.08410) (Sept 2026) combined steering vectors successfully when each went in at its own best layer and strength. Nearly perpendicular pairs combined best. Earlier work found summed safety vectors interfered destructively.
- **Dynamic composition exists only for dense vectors.** Scalena et al. (2024) combined several properties with "dynamic activation composition", but with dense vectors rather than neurons or personas.

### Measurement pitfalls to design around

- **Batch effects.** Under strong pushes, about a third of replies change when only the batch changes.
- **Judges.** In McKenzie et al., a broad judge rated "gradual recovery" 8.2 out of 10 where humans gave 3.3. Phrase-based refusal checks scored 0.68 accuracy against 0.86 for a guard model in Yang et al.
- **Random controls.** Large random pushes can tip refusals, so every effect needs matched random-neuron controls.
- **Firing is not causing.** The evil neuron ranked 35,436th by activation on evil text.
- **Long-text coherence.** Simple word-variety scores misread long fluent text as collapsed past about 128 tokens.

### Style and fixing wrong answers

- **Style may combine cleanly.** In Liu's work, "register" controls, which change the mode a reply is written in rather than which behavior happens, combined even when simply stacked, unlike behavior switches.
- **Ready style candidates.** The persona library also lists 8,827 replies where the model stayed an assistant but rated 6 or more for weirdness. hedwigai found single Poet and Flatterer neurons, but no humor neuron.
- **Right answers can hide inside.** [Orgad et al.](https://arxiv.org/abs/2410.02707) (ICLR 2025) found models may encode the correct answer internally yet consistently write a wrong one, and that error detectors trained on one dataset do not transfer to others.
- **Narrow repairs already work.** Injecting an honesty direction during generation lifted honest answers from 0.00–0.07 to 0.77–0.99, even against an instruction to keep deceiving ([Glover](https://jaredglover.substack.com/p/caught-and-corrected-steering-language)). Qwen-Scope stopped unwanted Chinese–English switching by suppressing one feature.
- **Steering is not a free upgrade.** AxBench and an honesty study found prompting and fine-tuning beat activation steering, and naive steering degrades coherence and capability ([summary](https://www.lesswrong.com/posts/Rpq28FTgPMXGHt9eD/restoring-model-alignment-via-honesty-activation-steering-1)).
- **Skill switches exist, but not for everything.** Liu's operator neurons turn addition into multiplication while keeping task skill inside their dose window, yet Liu's searches found no single neuron that turns prose into valid JSON.

### Gaps this program fills

1. How onset time and push duration shape neuron effects, including whether a short push spares capability.
2. How persona neurons combine at scale, and whether geometry predicts the result.
3. How combinations behave over time: handoffs, order and alternation.
4. Whether models notice, correct or explain persona-level tampering.
5. How persona steering interacts with visible reasoning.
6. Whether neurons, sparse-autoencoder features and dense vectors follow the same timing rules.
7. Whether pushes can change writing style while keeping meaning, and whether style combines more cleanly than persona.
8. Whether targeted pushes can fix systematic wrong answers on held-out questions without breaking right ones.

## Prerequisite knowledge

You need working knowledge in four areas before Phase 1: how a transformer carries information, how to intercept it in code, how to design a controlled experiment, and how to score outputs at scale. Plan on about four weeks of study, run alongside Phase 0.

### Concepts

- **Tokens and chat templates.** Every prompt is wrapped in a template with special markers and an optional "thinking" block. Template mistakes create fake effects, so the template is part of the experiment.
- **Residual stream.** Each token carries a vector of numbers (2,560 in Qwen3.5-4B, 4,096 in Qwen3-8B) from layer to layer. Every layer reads it and adds to it.
- **MLP neurons.** Inside each layer, Qwen's feed-forward block computes 9,216 (Qwen3.5-4B) or 12,288 (Qwen3-8B) intermediate values. Each value writes into the residual stream through its own column of the `down_proj` matrix, its "write direction". Pushing a neuron means adding a number to that value.
- **Normalization.** Layers read the direction of the residual stream, not its size. That is why doses are measured relative to the residual's typical size, and why too large a push drowns everything else out.
- **Prompt pass and generation steps.** The prompt is processed in one pass, then the reply one token at a time. Past tokens are stored in a cache, so a push on early tokens lives on in that cache after the push stops.
- **Decoding.** Greedy decoding always picks the most likely token; sampling adds randomness controlled by temperature and a seed. Batch size can also change outputs under strong pushes.
- **Units of intervention.** Raw neurons, sparse-autoencoder features and dense steering vectors are three ways to pick a direction to push. Neurons often do several jobs at once (polysemanticity).
- **Geometry.** The cosine between two neurons' write directions says whether they push the same way (near 1), opposite ways (near −1) or unrelated ways (near 0).

The dose rule used by both the evil recipe and the Qwen3-8B library sets the added amount c for neuron n at layer L from a dose α, the median residual size r̃ at that layer, and the length of the neuron's write direction:

```latex
c = \alpha \cdot \frac{\tilde{r}_L}{\lVert W_{\text{down}}[:, n] \rVert}
```

### Skills

- **Python, notebooks and Git.** Enough to read, run and version other people's code.
- **PyTorch hooks.** A hook on a layer's `down_proj` input lets you add to one neuron's value at chosen tokens. A token counter inside the hook turns a constant push into a schedule.
- **Hugging Face transformers.** Loading a model, applying its chat template, and generating token by token.
- **GPU workflow.** Renting a cloud GPU, moving files, and saving every run as structured records (prompt, schedule, seed, output, scores).
- **AI-assisted coding.** Claude Code can write and review the hooks and pipelines; your job is to check each result against a known effect before trusting it.

### Methods

- **Controls.** An unsteered baseline, random neurons matched in layer and dose, and "sham" schedules that push for the same number of tokens at the wrong time.
- **Pre-registration.** Write each hypothesis, measure and analysis down before running it, and keep a dated log.
- **Statistics.** Proportions with Wilson confidence intervals, bootstrap intervals, models with prompts as a random effect, and false-discovery-rate control when testing thousands of pairs.
- **Interaction scoring.** Judge a combination against what the two parts would give if they acted independently (the Bliss independence rule from pharmacology): above it is synergy, below it antagonism.
- **Judge validation.** Hand-label a random sample of outputs and report how often each automatic judge agrees with you before using it at scale.

### Four-week learning path

| Week | Focus | Done when |
| --- | --- | --- |
| 1 | Transformer concepts, Python and PyTorch basics | You can explain the residual stream and run Qwen3.5-4B on a prompt |
| 2 | Hooks and token-by-token generation | You can push one neuron and see the reply change |
| 3 | Experimental design and statistics | You can run a matched random control and compute a Wilson interval |
| 4 | Judges and text embeddings | You can score 100 replies automatically and compare with your own labels |

Core reading, in order: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), [hedwigai's neuron write-up](https://www.hedwigai.com/blog/qwen-personality-neurons), [Liu (2026)](https://arxiv.org/abs/2606.19831), [The Assistant Axis](https://www.anthropic.com/research/assistant-axis), [McKenzie et al. (2026)](https://arxiv.org/abs/2602.06941) and [Kang et al. (2026)](https://arxiv.org/abs/2609.08410).

### For the later phases

- **Style measurement.** Style can be counted (sentence length, vocabulary variety, reading level) or scored by a formality classifier, and must be kept separate from meaning.
- **Meaning comparison.** Text embeddings and entailment models judge whether two replies say the same thing in different words.
- **Probing.** A simple classifier trained on internal activations can show what the model "knows" even when it writes something else.
- **Honest search.** When thousands of candidates are tried, some fix answers by chance. Split questions into search and held-out halves, and compare against an equally large search over random units.
- **Adapters.** Small trainable add-ons (LoRA-style adapters) are needed for the push-versus-training pilot.

## Research categories

The program splits into nine tracks. Tracks A to C are the core, D and E ask what the model notices, F and G test how far results reach and what they mean for safety, and H and I close the program with style and the repair challenge. A scored backlog of 25 further ideas follows the tracks. Each question has an ID used in the phases, and each prediction is written down before the matching experiment runs.

| Track | Core question | Main phase |
| --- | --- | --- |
| A. Timing | When and for how long must a neuron be pushed? | 1 |
| B. Combinations | What happens when several neurons are pushed at once? | 2 |
| C. Choreography | What happens when pushes start and stop at different times? | 3 |
| D. Self-monitoring | Does the model notice, correct or explain the tampering? | 4 |
| E. Reasoning | How do pushes interact with visible thinking? | 4 |
| F. Generalization | Do results hold across models, prompts and intervention types? | 5 |
| G. Safety and forensics | Can pushes be detected or neutralized? | 5 |
| H. Style | Can pushes change how it writes without changing what it says? | 7 |
| Backlog | Which of 25 further ideas show a signal in one-week pilots? | 8 |
| I. Repair | Can pushes fix answers it usually gets wrong without breaking others? | 9 |

### A. Timing of single neurons

- **A1 Onset.** Switch a neuron on at reply token 0, 10, 30 or 100. Does the effect start at once, at the next sentence, or not at all?
- **A2 Kick and release.** Push only the first 1, 4, 16 or 64 reply tokens. How long does the effect last after release, and does capability survive?
- **A3 What carries it.** After a pulse, continue three ways: with the pushed cache, with the cache rebuilt from the same words without the push, and with the words swapped for neutral text. This separates the text from hidden memory.
- **A4 Sensitivity map.** Fire the same short pulse at every position. Are first tokens, sentence starts or line breaks where decisions are made?
- **A5 Dose versus duration.** Does a brief strong pulse equal a long weak one with the same total push?
- **A6 Prompt versus reply.** Push only while reading the prompt, only while writing the reply, or both.

Predictions: persona effects outlast the push for longer when the pulse is longer, because the text carries them (as Liu found for language routing). A pulse of 16 tokens or fewer keeps at least 70% of the full-time persona effect while capability stays within 10 points of baseline. Rebuilding the cache without the push keeps over 80% of the effect.

### B. Combinations

- **B1 Outcome classes.** Two neurons pushed together give a blend, one dominating, cancellation, a new persona, or collapse. How often each?
- **B2 Geometry.** Does the cosine between write directions, plus layer distance, predict the outcome class?
- **B3 Budget.** How many neurons can be pushed before collapse, and does splitting the dose stretch that limit?
- **B4 Redundancy.** Do two neurons with the same persona label stack into a stronger version or clash?
- **B5 Pecking order.** Is there a consistent dominance order across persona categories, and does it follow layer depth or distance from the Assistant Axis?
- **B6 Antidotes.** Do some neurons cancel many personas at once, and do they point toward the Assistant end of the axis?
- **B7 Emergence and recipes.** Which sets of 3 or 4 neurons create a character none of the parts makes alone? Can a search find a recipe for a requested character?

Predictions: near-perpendicular pairs blend, opposed pairs cancel, and aligned pairs reinforce. Collapse follows the combined push size, so for perpendicular pairs in one layer it begins once the square root of the summed squared doses passes about 1.65 in Qwen3.5-4B. The persona farther from the Assistant end wins dominance contests.

### C. Choreography over time

- **C1 Handoffs.** Push A for k tokens, then B. Is the switch clean, blended, or a hybrid, and is there a burst of incoherence at the switch?
- **C2 Order and lock-in.** Compare A then B with B then A. Does the first persona lock in, or does the newest push win? How long must A run before it survives B?
- **C3 Alternation.** Swap A and B every 1, 2, 4, 8, 16 or 32 tokens. At what speed do they fuse into one hybrid instead of taking turns?
- **C4 Overlap.** Keep A on throughout and pulse B. Does B break through, and for how long?
- **C5 Persona memory.** After a switch to B, does B continue, acknowledge or deny what A said?
- **C6 Return home.** After all pushes end, how fast does the model return to its normal Assistant voice?

Predictions: the first persona wins once it has run past its lock-in time. Alternation fuses below a period of a few words and alternates above a sentence.

### D. Self-monitoring and self-report

- **D1 Spontaneous correction.** How often does the model say something like "wait, that's not right" under or after a persona push, compared with off-topic topic pushes?
- **D2 "Why did you say that?"** After a steered passage, ask for an explanation. Score whether it notices tampering, owns the words, invents a reason, or denies saying them. Compare with the same text inserted without any push.
- **D3 Direct detection.** Ask whether anything unusual happened, with unsteered runs to measure false alarms.
- **D4 Identity reports.** Ask "Who are you?" at several points in a choreography. Does the answer track the active push or the text so far?

Predictions: persona pushes trigger self-correction less often than topic pushes, because McKenzie et al. found abstract steering harder to notice. Explanations of steered text look the same as explanations of inserted text, meaning they come from reading the words rather than from introspection.

### E. Reasoning

- **E1 Thinking versus answer.** Push only during the thinking block, or only during the answer.
- **E2 Repair.** Does visible reasoning pull a pushed answer back toward the normal assistant?
- **E3 Early warning.** Does the persona show up in the thinking before the answer, so a reader of the thinking could catch it?

Predictions: a persona pushed only during thinking carries partly into the answer, echoing the 48% carry-over Yang et al. found for refusal. Thinking mode weakens answer-only pushes.

### F. Generalization

- **F1 Models.** Qwen3.5-4B and Qwen3-8B first, then other Qwen3.5 sizes, Llama-3.1-8B, and Gemma as a contrast where single neurons cannot steer.
- **F2 Units.** Run the same schedules with raw neurons, Qwen-Scope sparse-autoencoder features and dense persona vectors.
- **F3 Conditions.** New prompt types, sampling instead of greedy decoding, and 8-bit instead of full precision.

Predictions: larger models recover faster and self-correct more. Onset and persistence follow the same curves across units once doses are scaled to the residual size.

### G. Safety and forensics

- **G1 Detection.** Can a simple probe on later layers, or a classifier on the text, tell that a push happened and when it started?
- **G2 Capping as a defense.** Does capping activity along the model's Assistant Axis cancel neuron-made personas, including their afterglow once the push stops?
- **G3 Prompt defenses.** Does a self-monitoring instruction raise self-correction under persona pushes, as it did for topic pushes?

Predictions: a probe detects pushes reliably at the layer after injection, while text-only detection lags the onset. Capping removes most single-neuron persona effects at a small capability cost.

### H. Style

- **H1 Style dials.** Find units that change formality, length, reading level, tone or a poetic voice, and measure their dose curves.
- **H2 Meaning kept.** Does the reply say the same thing in a new style? The meaning score against the unsteered reply must stay above a set threshold while style measures move.
- **H3 Voice versus identity.** Does a "poet" unit change only the voice, or also make the model claim to be a poet?
- **H4 Style combinations.** Do style units stack more cleanly than persona units, as Liu's register result suggests? Test style with style and style with persona.
- **H5 Style over time.** Style handoffs mid-reply, persistence after release, and alternation.
- **H6 Fingerprints.** Can a classifier tell from the text alone which style unit was pushed?

Predictions: style units combine more cleanly than persona units. Meaning holds at moderate doses and breaks near the collapse point. Style persists after release at least as long as persona, because the text itself carries it.

### I. Repair: fixing wrong answers (the final challenge)

- **I1 Failure bank.** Questions the model reliably gets wrong across seeds: arithmetic slips, letter counting, common misconceptions, logic puzzles, factual recall and unwanted language mixing.
- **I2 Hidden knowledge.** For which failures does the right answer already show up inside, through a probe at the answer tokens or a high rank among the model's next-token choices?
- **I3 Search.** Find pushes that fix failures on a search half, comparing neurons, sparse-autoencoder features and dense vectors, and full-reply pushes against pulses only while the answer is written.
- **I4 Held-out test.** Measure the fix rate on unseen failures of the same type, against an equally large search over random units.
- **I5 Do no harm.** Accuracy on questions it already answers correctly must not drop. Report net gain: answers fixed minus answers broken.
- **I6 Error types.** Which kinds of errors are fixable, such as systematic slips, and which are not, such as missing knowledge?

Predictions: fixes work for narrow, systematic errors where the right answer is already inside, and fail where knowledge is missing. Answer-time pulses beat full-reply pushes on net gain. No single unit gives a general accuracy boost.

## Exploration backlog

Twenty-five further ideas wait in a scored backlog. Each has a first-pass Value score (what it would teach) and Cost score (effort beyond the existing pipeline), both from 1 to 5. Phase 8 pilots the best value-for-cost ideas and promotes those that show a signal.

- **Value:** 5 would change how steering is understood; 1 is a curiosity.
- **Cost:** 1 runs on the existing pipeline with only a new experiment file; 5 needs new tooling or training.
- **Pilot rule:** about 200 runs in one week, with matched random controls and a success margin written down before the run.
- **Status:** update it in the table as ideas move through pilots.

| ID | Group | Idea | Value | Cost | Status |
| --- | --- | --- | --- | --- | --- |
| X1 | Knowledge | Temporary fact swaps: make it insist Paris is in Italy, then measure recovery after release | 4 | 2 | Backlog |
| X2 | Knowledge | Topic magnets: force one subject into every reply | 3 | 1 | Backlog |
| X3 | Knowledge | Topic blockers: make one subject impossible to discuss | 4 | 3 | Backlog |
| X4 | Knowledge | Confidence dial: does more hedging line up with being wrong? | 4 | 2 | Backlog |
| X5 | Reasoning | Planning moments: push before a rhyme or a solution's first step to see when plans lock in | 5 | 2 | Backlog |
| X6 | Reasoning | Thinking length: make thinking shorter or longer and track accuracy | 4 | 2 | Backlog |
| X7 | Reasoning | Strategy switches: find operator-style switches for other skills | 4 | 3 | Backlog |
| X8 | Emotion | Emotion neurons (joy, anger, fear) and whether they leak into later turns | 3 | 2 | Backlog |
| X9 | Social | Sycophancy and flattery dials, starting from hedwigai's Flatterer neuron | 4 | 2 | Backlog |
| X10 | Social | Humor hunt: is humor spread across many neurons? | 3 | 3 | Backlog |
| X11 | Language | Language and dialect control, including stopping unwanted language mixing | 3 | 1 | Backlog |
| X12 | Format | Length, lists and JSON: is there a multi-neuron JSON switch? | 3 | 3 | Backlog |
| X13 | Conversation | Persona echo: push in turn 1, measure turn 10 | 4 | 2 | Backlog |
| X14 | Conversation | Two steered models talking: does one persona spread to the other? | 4 | 3 | Backlog |
| X15 | Conversation | Self-image: does it treat its own steered words as evidence of who it is? | 4 | 2 | Backlog |
| X16 | Self-knowledge | Name the neuron: pick the pushed neuron's label from a menu | 4 | 1 | Backlog |
| X17 | Self-knowledge | Predict the push: forecast its own change before the push happens | 4 | 2 | Backlog |
| X18 | Self-knowledge | Identity limit: how far along the Assistant Axis before identity breaks? | 3 | 2 | Backlog |
| X19 | Mapping | Opposites: are a neuron's up and down personas opposites? | 3 | 1 | Backlog |
| X20 | Mapping | Persona map: cluster thousands of neurons by effect | 4 | 3 | Backlog |
| X21 | Mapping | Effect shapes: which effects act as switches, dials or cliffs? | 3 | 1 | Backlog |
| X22 | Mapping | Base versus chat models: do persona neurons exist before chat training? | 5 | 3 | Backlog |
| X23 | Robustness | Survival: sampling temperature, 4-bit models and very long replies | 3 | 1 | Backlog |
| X24 | Training | Push versus training: train an adapter to mimic a push and compare | 4 | 4 | Backlog |
| X25 | Training | Fine-tuning: does it move or erase persona neurons? | 4 | 4 | Backlog |

## Experimental toolkit

Every experiment is one record: a model, one or more interventions with a dose and a schedule, a prompt, a decoding setting and a seed, scored by a fixed set of measures. Standardizing that record lets all nine tracks and the backlog pilots share one pipeline and one growing dataset.

### Intervention units

| Unit | Source | Used for |
| --- | --- | --- |
| Evil pair and "zoo" neurons, Qwen3.5-4B | lyra, hedwigai, Bowling | Phase 0 replication, Track A |
| Persona library neurons, Qwen3-8B | Neuron Persona Library export | Tracks B and C |
| Behavior gates: language routing (Qwen3-4B L25 N347), arithmetic operator (Qwen3.5-2B L16 N2493) | Liu (2026) | Timing calibration with automatic scoring |
| Self-found neurons | Your own screen with neuron-explorer | Filling missing persona categories |
| Sparse-autoencoder features | Qwen-Scope | Track F2 |
| Dense persona vectors | Contrastive prompts, persona-vector method | Track F2 |
| Assistant Axis direction | Re-derived for each model with the published method | Tracks B6 and G2 |

### Schedules

A schedule lists segments, each with a unit, sign, dose, start token and end token. Token 0 is the first reply token; "P" means the prompt pass. Common shapes:

- **Constant:** on from P to the end (the viral setup, used as the reference).
- **Pulse:** on for tokens 0 to 15, then off.
- **Late onset:** on from token 50 to the end.
- **Handoff:** A on tokens 0 to 39, B from 40 to the end.
- **Alternation:** A and B swap every m tokens.
- **Ramp:** dose rises from 0 to full over 20 tokens, to test gradual against sudden change.

Runs are stored one per line, for example:

```csv
run_id,model,unit,layer,index,sign,dose,start,end,prompt_id,seed
A2-0001,Qwen3.5-4B,neuron,14,8949,+,1.0,0,16,story-07,0
A2-0002,Qwen3.5-4B,neuron,14,8949,+,1.0,0,64,story-07,0
A2-0003,Qwen3.5-4B,random,14,3120,+,1.0,0,16,story-07,0
C1-0001,Qwen3-8B,neuron,18,7807,+,0.75,0,40,story-07,0
C1-0001,Qwen3-8B,neuron,22,1934,+,0.75,40,end,story-07,0
```

(Neuron indices in the last two rows are placeholders.)

### Dosing rules

- **Scale:** doses are in units of the layer's median residual size, the same rule as the evil recipe and the persona library.
- **Singles:** measure each unit's lowest reliable dose with a ladder from 0.25 to 1.5, and its collapse dose.
- **Combinations:** keep the combined push, the square root of the summed squared doses, below the model's collapse ceiling measured in Phase 0. Compare equal splits with doses tuned per unit.
- **Controls:** every condition gets random neurons matched in layer, dose and schedule.

### Prompt sets

- **Identity probe:** "Who are you?", matching the persona library.
- **Long-form set:** 40 story, explanation and advice prompts that draw 300 to 600-token replies, needed for timing work.
- **Capability set:** about 100 short questions with exact answers (arithmetic, facts, simple reasoning), scored by exact match.
- **Self-report probes:** follow-up turns such as "Why did you say that?" and "Did anything unusual happen?"
- **Held-out set:** prompts never used for discovery or tuning.
- **Excluded:** requests for harmful content, in every track.

### Controls

- An unsteered run for every prompt and seed.
- At least 5 random neurons per condition, matched in layer, dose and schedule.
- Sham timing: the same pulse length at a different position.
- Text-only control: the steered text inserted with no push (questions A3 and D2).
- Batch control: fixed batch size and order, with 5% of runs repeated in a second layout.

### Measures

| Measure | What it captures | How |
| --- | --- | --- |
| Persona timeline | Which persona is "speaking" in each sentence | Text-embedding similarity to each persona's reference replies, checked against an AI judge and hand labels |
| Onset and offset | When the effect starts and ends | First and last sentence above a set threshold |
| Half-life | How long an effect lasts after release | Tokens until the persona score falls to half its peak |
| Divergence | How far the model moved from normal | Per-token difference between steered and unsteered next-token probabilities on the same text, and the first token that differs |
| Capability | Whether the model can still do tasks | Exact-match accuracy on the capability set under the same schedule |
| Coherence | Whether text stays readable | Perplexity under the unsteered model, repetition rate, and an AI judge for long texts |
| Self-correction | Explicit "wait, that's not right" moments | Judge that splits replies into attempts, following McKenzie et al. |
| Explanation class | What the model says about steered text | Four-class judge rubric, validated on 200 hand labels |
| Internal readouts | What happens inside after release | Per-token projection on the neuron's write direction and on the Assistant Axis |
| Interaction score | Synergy or antagonism in a pair | Observed effect minus the effect expected if the two acted independently |

### Statistics and records

- **Sample size:** 96 prompts per condition gives proportions to within about ±10 points (95% Wilson interval).
- **Models:** logistic mixed-effects models with prompt and neuron as random effects.
- **Many tests:** false-discovery-rate control at 5% across pair screens.
- **Pre-registration:** hypotheses and analyses in a dated file in the repository; anything else is labeled exploratory.
- **Reproducibility:** fixed seeds, model revision hashes, library versions and GPU type saved with every run; raw outputs kept.

### Additions for style, repair and exploration

- **Units:** style candidates from the library's changed-assistant replies, hedwigai's Poet and Flatterer neurons, and screens for missing styles; repair candidates from screens that contrast failed and correct answers.
- **Schedules:** an answer-time pulse that pushes only while the answer itself is written, and turn-scoped pushes for multi-turn conversations.
- **Prompt sets:** a style set of 40 prompts whose meaning can be checked; a failure bank split into search and held-out halves; 1,000 correctly answered questions for harm checks; 10-turn conversation scripts; openers for two-model dialogues; and label menus for self-knowledge probes.
- **Measures:** style scores (sentence length, vocabulary variety, reading level, formality); meaning kept (embedding similarity and entailment against the unsteered reply); repair scores (fix rate, harm rate and net gain); hidden-knowledge probe accuracy; and persona echo across turns.
- **Controls:** an equally large search over random units for every repair search, held-out evaluation only, and paraphrase pairs to calibrate the meaning score.

## Phases

The plan runs 49 weeks at roughly 12 to 15 hours a week, in eleven phases that each end at a go/no-go gate. Nothing new runs until Phase 0 reproduces three known effects, because every later result depends on the push engine being right.

*(Diagram: research roadmap · 11 phases, 10 gates. See the Claude Doc version of this plan to view it.)*

A failed gate means fixing the pipeline or reporting a null result, never skipping ahead to the next phase.

### Phase 0: Foundations and replication (weeks 1 to 4)

- **Learn:** the four-week path under Prerequisite knowledge.
- **Build:** the push engine (hooks that follow a schedule), the run record format, and the scoring pipeline.
- **Replicate:** L14 N8949 hostility on 20 neutral prompts (target at least 14 of 20, against near zero for random neurons); Liu's language-routing neuron flipping on the first token and persisting after release (target at least 0.8); and 20 Qwen3-8B library personas on "Who are you?" (target at least 85% reproduced).
- **Calibrate:** each model's collapse ceiling with a dose ladder; judge agreement on 200 hand-labeled replies.
- **Gate 0:** all three replications pass, and batch-control reruns agree at least 90% of the time.

### Phase 1: Timing of single neurons (weeks 5 to 8, Track A)

- **Units:** 20 validated units: the evil neuron, three zoo personas, the language and arithmetic gates, and 14 library personas spread across layers.
- **Runs:** questions A1 to A6, about 12 schedules per unit on 40 long-form prompts; about 15,000 runs with controls.
- **Output:** onset, half-life and commitment point per unit, and a direct answer to whether a short pulse spares capability.
- **Gate 1:** at least one timing pattern holds with confidence intervals that exclude no effect. If none does, Phase 2 still runs and the null result is written up.

### Phase 2: Curation and combinations (weeks 9 to 13, Track B)

- **Curate:** keep one neuron per persona name, drop low-coherence entries, re-test on long prompts, and measure each survivor's minimum dose. Target 200 to 300 robust personas.
- **Geometry pass:** cosines and Assistant-Axis positions for every pair, with no text generated.
- **Pair screen:** 2,000 to 3,000 pairs sampled across cosine and layer-distance bins, 8 prompts each, plus same-name pairs, budget tests with up to 6 neurons, antidote search, and triple searches around synergistic pairs. About 25,000 runs.
- **Gate 2:** outcome classes agree at least 80% on held-out prompts, and the geometry test gives either a clear predictor or a clear negative.

### Phase 3: Choreography (weeks 14 to 17, Track C)

- **Pairs:** about 50 pairs from Phase 2 covering blends, dominance and new personas.
- **Runs:** handoffs at 10, 20, 40 and 80 tokens, both orders, alternation every 1 to 32 tokens, overlaps, persona-memory probes and return-home timing. About 15,000 runs.
- **Output:** lock-in times, the alternation speed where personas fuse, a taxonomy of handoffs.
- **Gate 3:** at least one schedule effect replicates in both Qwen3.5-4B and Qwen3-8B.

### Phase 4: Self-monitoring and reasoning (weeks 18 to 21, Tracks D and E)

- **Runs:** self-correction, explanation, detection and identity probes on saved Phase 1 and 3 runs plus about 6,000 new runs, including text-only controls. Thinking-mode experiments on Qwen3-8B.
- **Labels:** 300 hand-labeled explanations to validate the four-class judge.
- **Gate 4:** the explanation judge agrees with hand labels at Cohen's kappa of at least 0.7.

### Phase 5: Generalization and safety (weeks 22 to 24, Tracks F and G)

- **Runs:** the core timing, pair and handoff experiments on two more Qwen3.5 sizes, Llama-3.1-8B and a Gemma contrast; neurons against sparse-autoencoder features and dense vectors; sampling and 8-bit; detection probes, Assistant-Axis capping and self-monitoring prompts. About 15,000 runs.
- **Gate 5:** every headline finding is labeled general, model-specific or unit-specific.

### Phase 6: First release (weeks 25 to 27)

- **Write:** the report, with every number tied to its run records and pre-registered or exploratory status.
- **Release:** code, the full run dataset, and an interactive explorer showing persona timelines and the interaction map.
- **Review:** drafts to two or three outside readers, then a public post and an arXiv or workshop submission.

### Phase 7: Style (weeks 28 to 32, Track H)

- **Units:** style candidates from the library's changed-assistant replies, hedwigai's Poet and Flatterer neurons, and screens for missing styles such as formal, casual, terse and simple.
- **Runs:** questions H1 to H6: dose curves, meaning-kept tests, voice-versus-identity probes, style pairs and style-persona pairs, and style handoffs. About 12,000 runs.
- **Gate 7:** at least three style units move their style score while the meaning score stays above threshold on held-out prompts.

### Phase 8: Exploration sprints (weeks 33 to 38)

- **Pick:** the backlog ideas with the best Value for their Cost, about two a week.
- **Pilot:** about 200 runs each, with matched random controls and a success margin set in advance.
- **Promote:** ideas that show a signal grow into full studies in the remaining weeks or the final report; the rest are written up as short notes. About 20,000 runs in total.
- **Gate 8:** at least two pilots promoted, and every pilot's result, positive or null, logged.

### Phase 9: Repair, the final challenge (weeks 39 to 46, Track I)

- **Failure bank:** about 1,000 questions across six error types, keeping those the model gets wrong in at least 4 of 5 sampled attempts, split into search and held-out halves.
- **Hidden knowledge:** probe every failure for a right answer already inside.
- **Search and test:** search on one half, test on the held-out half against an equally large random search, and check harm on 1,000 questions it already answers correctly. About 40,000 short runs plus candidate screens.
- **Gate 9:** a repair effect that survives the held-out test and the random-search comparison, or a clearly documented null.

### Phase 10: Final synthesis and release (weeks 47 to 49)

- **Write:** the full report across all nine tracks and the backlog.
- **Release:** version 2.0 of the code, dataset and explorer, with style and repair views.
- **Review:** the same outside-reader and publication route as Phase 6.

## Compute, tools and budget

The whole program needs roughly 150,000 runs and about 275 GPU-hours, mostly on one rented 24 GB card, for an estimated $460 to $960 in total over 11 to 12 months. Rates below are [September 2026 list prices](https://www.hostinger.com/au/tutorials/cloud-gpu-pricing/): an RTX 4090 costs $0.34 an hour on RunPod's Community Cloud and $0.74 on its Secure Cloud, and an A100 80GB starts at $1.39.

| Phase | Runs (approx.) | GPU | GPU-hours (est.) |
| --- | --- | --- | --- |
| 0 Foundations | 5,000 | RTX 4090 | 20 |
| 1 Timing | 15,000 | RTX 4090 | 25 |
| 2 Combinations | 25,000 | RTX 4090, A100 for big batches | 40 |
| 3 Choreography | 15,000 | RTX 4090 | 25 |
| 4 Self-monitoring | 6,000 | RTX 4090 | 15 |
| 5 Generalization | 15,000 | A100 80GB | 30 |
| 6 First release | none | none | 5 |
| 7 Style | 12,000 | RTX 4090 | 20 |
| 8 Exploration | 20,000 | RTX 4090, A100 for training pilots | 40 |
| 9 Repair | 40,000 short runs plus screens | A100 80GB | 50 |
| 10 Final release | none | none | 5 |

Generation itself is about 48 million tokens, roughly 15 GPU-hours at a conservative 1,000 tokens per second with hooks. The rest covers local judge runs, scoring and divergence passes, debugging, reruns and idle time.

### Hardware

- **Main card:** a rented RTX 4090 (24 GB) holds Qwen3.5-4B and Qwen3-8B at full 16-bit precision with room for modest batches.
- **Bigger runs:** an A100 80GB for Phase 2 batches, the larger Phase 5 models, Phase 8 training pilots and the Phase 9 searches.
- **Local checks:** a Mac with 16 GB or more can run the 4B model in 8-bit with MLX, as hedwigai did, for quick checks. Keep the main pipeline on one setup so numbers stay comparable.
- **Habits:** stop pods when idle, since idle time bills at the full rate, and keep data in cheap storage between sessions.

### Software

- **Core:** Python, PyTorch and Hugging Face transformers for scheduled pushes; vLLM only for fast unsteered baselines, because custom per-token hooks are harder there.
- **Interpretability:** lyra's neuron-explorer for discovery, Qwen-Scope sparse autoencoders for Track F2, and optionally nnsight for cleaner interventions.
- **Scoring:** an open text-embedding model for persona timelines; a local judge model from a different family than the model under test for bulk scoring; a stronger API judge only for validation samples.
- **Data:** one Parquet file per experiment and DuckDB for queries; experiment tracking in Weights & Biases or a plain run log.
- **Coding help:** Claude Code to write and review hooks, tests and analysis notebooks.
- **Access:** Llama-3.1-8B requires accepting Meta's license on Hugging Face before Phase 5.

### Budget estimate

| Item | Estimate |
| --- | --- |
| GPU time (about 175 hours on RTX 4090, 100 on A100) | $200 to $350 |
| Storage for weights and outputs over 12 months | $120 to $240 |
| API judging of validation samples | $80 to $250 |
| Contingency for reruns | $60 to $120 |
| **Total** | **$460 to $960** |

The API line is a placeholder until judge models and sample sizes are fixed in Phase 0.

## Risks, ethics and safety

The biggest risk is a false finding: an effect that comes from a weak judge, a batch quirk or generic push damage rather than from the neuron and its timing. Every risk below has a control already built into the toolkit or the gates.

### Scientific and practical risks

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Generic push damage | Large random pushes can tip behavior too | Matched random neurons in every condition; report effect relative to them |
| Judge errors | Broad judges overscored "recovery" against humans | Hand-labeled validation, two judges, agreement reported; narrow rubrics |
| Batch and precision quirks | A third of strongly pushed replies change with batch layout | Fixed layouts, 5% reruns in a second layout, precision recorded |
| Chat-template artifacts | Template choices can create fake steering effects | One fixed template per model; thinking on and off tested separately |
| False positives across thousands of pairs | Some pairs look special by chance | False-discovery-rate control and held-out replication before any claim |
| Unreliable library labels | One prompt, automatic labels | Re-test on long prompts; hand-check 100 personas in Phase 2 |
| Scope creep | The combination space is effectively infinite | Fixed run budgets per phase; gates decide what continues |
| Null results | Persistence or geometry may show nothing | Pre-registered nulls are still reported and published |
| Learning curve | The work needs new coding and statistics skills | Four-week path, replications first, AI-assisted code review |
| Moving targets | New Qwen versions and library updates | Pin model revisions and library versions per run |

### Ethics and safety

- **No harmful requests.** No track uses prompts asking for dangerous help. Hostile personas are studied only on neutral prompts, as hedwigai did.
- **No new jailbreak recipes.** Single-neuron safety bypass is already [published](https://arxiv.org/abs/2605.08513) for open models. This program does not publish neuron recipes whose main effect is removing refusals. Any such finding is reported in aggregate and shared privately with the model's developer first.
- **Careful excerpts.** Hostile or disturbing outputs appear only as short excerpts with a content note; full transcripts stay in the dataset behind a clear label.
- **No live steered chatbot.** The public explorer shows recorded runs, not live generation.
- **No overclaiming.** Results describe behavior. A model "explaining" or "noticing" something is reported as text it produced, not as evidence of inner experience.
- **Licenses.** Check the license of every third-party dataset before redistributing; for example, Sauers' owl training data is non-commercial.

### Risks for the later phases

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Chance fixes in the repair search | Trying thousands of candidates guarantees some lucky fixes | Search and held-out halves, an equally large random search, and claims only from held-out results |
| Style edits that change meaning | A "style" effect could secretly be a content change | A meaning-score threshold set on paraphrase pairs before Phase 7 |
| Backlog sprawl | Twenty-five ideas could eat the year | One-week pilots, a pre-set success margin, and promotion only on a signal |
| Training pilots growing into projects | Adapter training needs new skills and compute | A 20 GPU-hour cap per training pilot; drop it if it overruns |

- **Fact swaps stay labeled.** Swapped-fact outputs are marked in the dataset and explorer and never presented as true.
- **Repair claims are held-out only.** No accuracy claim rests on questions used during the search.
- **Two-model conversations stay in the lab.** Steered models never talk to real users.

## Outputs and publication

The program releases four things in stages, an open push engine, a run dataset, an interactive explorer and a written report, so each phase gets outside feedback before the next builds on it.

| Output | What it contains | When |
| --- | --- | --- |
| Push engine | Open-source code for scheduled pushes on neurons, features and vectors, with tests that reproduce the three Phase 0 effects | Week 4 |
| Timing note | Short public write-up of Phase 1, including null results | Week 9 |
| Interaction map | Pair outcomes plotted against geometry, browsable online | Week 14 |
| Run dataset (version 1.0) | Every run with its schedule, output and scores, plus a data card | Week 27 |
| Explorer (version 1.0) | Persona timelines per run, the interaction map and a choreography viewer | Week 27 |
| Report (version 1.0) | Methods, pre-registered and exploratory results, limitations | Week 27 |
| Style note | Style dials, meaning-kept results and style combinations | Week 33 |
| Pilot notes | One short write-up per backlog pilot, positive or null | Weeks 33 to 38 |
| Repair report | Failure bank, search method, held-out results and harm checks | Week 47 |
| Version 2.0 | Full report, complete dataset, and an explorer with style and repair views | Week 49 |

### Where and how to publish

- **Milestone posts:** LessWrong and the AI Alignment Forum, with code on GitHub, where most of the October 2026 work appeared.
- **Early readers:** send drafts to two or three people whose work this builds on, such as the authors of the audits and the dose-law paper.
- **Formal venues:** an arXiv preprint, then a mechanistic interpretability workshop at a major machine learning conference; ICML 2026 hosted one, so check 2027 deadlines.

### What counts as done

- Every headline number traces back to run records anyone can download.
- At least one outside person reproduces a headline result within three months of release.

## Sources

**October 2026 neuron-steering work**

- [lyra, "evil qwen" recipe post](https://x.com/_lyraaaa_/status/2113833330429083662) (X, 24 Oct 2026)
- [hedwigai, Personality Neurons in Qwen3.5-4B](https://www.hedwigai.com/blog/qwen-personality-neurons)
- [Mr-Neutr0n, qwen3.5-4b-neuron-steering](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering)
- [occcat, qwen35-evil-neuron results](https://github.com/occcat/qwen35-evil-neuron/blob/main/RESULTS.md)
- [Martin Bowling, qwen-neuron-audit](https://github.com/martinbowling/qwen-neuron-audit)
- [Neuron Persona Library (Qwen3-8B)](https://qwen-neurons-site.vercel.app/)
- [treetown, Cookie Monster combination post](https://x.com/treetowntree/status/2114573606009328067)
- [Sauers, Qwen3-8B tawny owl (128 neurons)](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons)

**Papers**

- [Liu, Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering (2026)](https://arxiv.org/abs/2606.19831)
- [Kazemi, Chegini and Safi, A Single Neuron Is Sufficient to Bypass Safety Alignment (2026)](https://arxiv.org/abs/2605.08513)
- [Qi et al., Safety Alignment Should Be Made More Than Just a Few Tokens Deep (ICLR 2025)](https://arxiv.org/abs/2406.05946)
- [Seok and Kim, Generation Collapse in Contrastive Activation Addition Steering (ICML 2026 workshop)](https://openreview.net/forum?id=iRtgyVxRi1)
- [Yang et al., Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal (2026)](https://arxiv.org/abs/2605.26772)
- [McKenzie et al., Endogenous Resistance to Activation Steering in Language Models (ICML 2026)](https://arxiv.org/abs/2602.06941)
- [Lindsey, Emergent Introspective Awareness in Large Language Models (2025)](https://transformer-circuits.pub/2025/introspection/index.html)
- [Mechanisms of Introspective Awareness (2026)](https://arxiv.org/abs/2603.21396)
- [Lu et al., The Assistant Axis (Anthropic, Jan 2026)](https://www.anthropic.com/research/assistant-axis) and [paper](https://arxiv.org/abs/2601.10387)
- [Chen et al., Persona Vectors (2025)](https://arxiv.org/abs/2507.21509)
- [Qwen-Scope (2026)](https://arxiv.org/abs/2605.11887)
- [Kang et al., Compositional Multilingual and Behavioral Attribute Steering (2026)](https://arxiv.org/abs/2609.08410)
- [Elhage et al., A Mathematical Framework for Transformer Circuits (2021)](https://transformer-circuits.pub/2021/framework/index.html)
- [Johansson et al., choice blindness study coverage (Science, 2005)](https://innovations-report.com/social-sciences/report-50074)

**Costs**

- [Hostinger, Cloud GPU pricing compared (updated 17 Oct 2026)](https://www.hostinger.com/au/tutorials/cloud-gpu-pricing/)

**Style and repair**

- [Orgad et al., LLMs Know More Than They Show (ICLR 2025)](https://arxiv.org/abs/2410.02707)
- [Glover, Caught and Corrected: Steering Language Models Back to Honesty](https://jaredglover.substack.com/p/caught-and-corrected-steering-language)
- [Restoring model alignment via honesty activation steering (LessWrong)](https://www.lesswrong.com/posts/Rpq28FTgPMXGHt9eD/restoring-model-alignment-via-honesty-activation-steering-1)
