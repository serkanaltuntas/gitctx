# Joint likelihood over lossless windows

`gitctx.window_mixture` is an explicit aggregation alternative to individually
supervised partial-window messages. It keeps the same shared decoder parameters,
8,192-token per-window context and 256-token answer/EOS reserve. No whole-commit
target is attached to a partial window as an independently supervised example.

For W windows and a shared output prefix, define the next-token probability as

```
p(y_t | all windows, y_<t) = sum_w p_model(y_t | window_w, y_<t) / W
L = -mean_t log p(y_t | all windows, y_<t)
```

This averages probabilities, not logits or independent window losses. A commit
contributes one target and one answer-token count, regardless of window count.
With one window, the objective is ordinary answer-masked causal cross entropy.
Only audited original references or reviewed open-teacher overlays may supply
training targets; input preparation and aggregation do not approve labels.

## Inputs and coverage

`prepare` consumes a complete source record and existing audited lossless
windows, or builds the same deterministic partition. It checks identity,
source/prompt hashes, interval coverage and the absence of per-window targets.
Original windows are copied rather than mutated. Full inputs retain the common
student prompt; partial inputs use an explicit aggregation instruction with the
same exact source fragments and repeated context. The revised token budget is
checked again. References, historical subjects and answer lengths never determine
input windows. A changed/missing/over-budget window fails explicitly.

## Training and decoding

`answer_log_probs` computes only the causally shifted answer-position logits,
including the final EOS target. Selective projection avoids allocating the full
prompt-by-vocabulary logit tensor and does not change model parameters.

`backward_joint` computes all window target log probabilities without retaining
activation graphs, then replays one window at a time. Its per-token gradient
weight is that window's target probability divided by the sum across windows.
Treating these weights as constants in the replay produces the exact gradient
of L, not a different weighted training objective. Forward values must reproduce
between passes; non-finite or mismatched scores fail. The caller owns gradient
clearing and optimizer updates and must discard partial gradients after failure.
A one-window group needs only one forward/backward pass.

`generate` combines next-token probabilities across every window before one
shared token/EOS decision. Every branch receives the same generated prefix.
CPU cache storage has a fixed byte budget; windows beyond it replay their whole
prefix. Reducing cache memory affects speed, not source coverage or probabilities.
There is no retrieval, fragment vote, silent window exclusion or input cropping.

## Verification and limits

Tests compare selective projection and gradients with the original full head,
the streamed joint gradient with a single autograd graph, one-window loss with
the existing causal mask, and cached decoding with complete-prefix replay.
They also test future-target isolation, window-order invariance, contribution
from later windows, EOS, cache-budget fallback, source preservation and rejection
of malformed inputs. Corpus-wide preparation and a separate readiness audit are
still required before choosing any student training run.

This architecture shares the generated prefix across windows but provides no
direct attention between source windows. Correct aggregation arithmetic does not
prove semantic quality or cross-window reasoning. Long-input evaluation must
measure factual grounding, omitted changes and invented changes after a separately
approved training run. Model size, training partitions and release gates remain
unchanged by this method.
