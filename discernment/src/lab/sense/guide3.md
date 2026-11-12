# Labelling guide v3: how a user reacted, in two steps

The feedback sense reads how a person reacted to something an AI assistant did for them. It reads the person,
never the work: it does not judge whether the answer was correct, only what the person's next move and words
show about the result. It must work for any task.

The sense is a gate for learning: the assistant learns only from reactions that clearly say the result was
good. When in doubt, it is not a yes. So the sense answers two questions, in order.

## What you see

- **Request:** what the person asked for (sometimes missing).
- **Next move:** what kind of turn came next: replied, new request, asked to adjust, repeated the request,
  left the session, said nothing.
- **Words:** what the person said next (may be empty).

## Step 1: is it a clear verdict on the result?

A reaction is **clear** when its words say how the person found the result itself: it works, it is what they
wanted, they are building on it, or it missed what they asked, or it is wrong.

A reaction is **not clear** when it does not say that. If it is not clear, the answer is **null** and step 2
does not matter. Not clear:

| Kind | Why it is not clear | Examples |
| --- | --- | --- |
| Thanks for the effort, not the result | Thanking someone for trying says nothing good about what they delivered; people say it when it did not work | "Thank you so much for trying", "Thanks for checking anyway", "No worries, I'll look elsewhere", "That's alright, thanks for your time" |
| Sarcasm | The words say the opposite of what is meant, and you cannot be sure which | "Wow, great job", "Oh perfect, just what I needed" after a miss |
| Indirect or ambiguous | It could be read either way | "ok...", "hmm", "interesting", "I see", "right" |
| Settling or hedging | Reluctant, does not say it is good | "I guess that'll do", "whatever, fine", "it is an outcome" |
| Silence or moving on without a word | Nothing about the result | no words at all, or the next, different request with no comment |
| Off topic, or words you do not understand | Cannot tell | an unrelated reply, task-specific words whose meaning is unclear |

Leaving the session, or saying thanks, is neither good nor bad on its own: the words decide. "Perfect, that's
it, bye!" is clear (a yes); "That's fine, thanks for trying, I'll head out" is not clear.

## Step 2: if it is clear, what score?

| Score | Meaning | Typical words |
| --- | --- | --- |
| 100 | Exactly right: explicit praise or real satisfaction with the result | "Perfect, exactly what I needed", "That's the one", "It works now, thanks!" |
| 80 | Acceptable, or good enough to build on | "Ok, that works", "Fine, I'll take it", "Great, now can you also...", "Make it shorter", "Same but in blue" |
| 20 | One part of the result missed what was asked; most of it was fine | "I wanted it in blue", "I said size 9", "I asked for under 100 words" |
| 0 | Wrong: rejects it, says it did not do what was asked, repeats the same request, gives up after a complaint | "No, that's not what I asked", "That's not my style at all", "Forget it, this is useless" |

**Building on it vs correcting it.** A new preference the request did not mention ("make it shorter", "use
camelCase", "same but in blue") is building on the result: **80**. Only when the words show the result missed
the request ("I said...", "I asked for...", "like I asked") is it **20** (one detail) or **0** (the whole thing).

| Reaction | Did the request ask for it? | Score |
| --- | --- | --- |
| "Can you rename the variables to camelCase?" | No: a new preference | 80 |
| "Use camelCase like I asked" | Yes: it was missed | 20 |
| "Make it shorter" | No length asked | 80 |
| "I asked for under 100 words" | Yes | 20 |

**Polite unhappiness.** "Thanks! Could you find me [the same thing again]?" repeats the request: clear, **0**.

## Examples across tasks

| Task | Next move | Words | Clear? | Score |
| --- | --- | --- | --- | --- |
| Coding | replied | "It works now, thanks!" | yes | 100 |
| Coding | replied | "Still getting the same error" | yes | 0 |
| Coding | asked to adjust | "Can you rename the variables to camelCase?" | yes | 80 |
| Coding | asked to adjust | "I said camelCase" | yes | 20 |
| Writing | replied | "Love it. Now a shorter version for Twitter?" | yes | 80 |
| Booking | new request | (nothing) | no | null |
| Booking | repeated the request | "I need a hotel in the centre, with parking" | yes | 0 |
| Recommending | replied | "Ooh, haven't seen that one, adding it" | yes | 100 |
| Shopping-like | left the session | "That's perfectly fine, thank you so much for trying! I'll look on my own" | no | null |
| Support | left the session | "Thanks, that fixed it. Bye!" | yes | 100 |
| Support | replied | "Thanks for checking anyway" | no | null |
| Search | replied | "hmm" | no | null |
| Any | replied | "Wow. Great job." (after a miss) | no | null |
| Any | replied | "I guess that's fine" | no | null |
| Any | said nothing | (nothing) | no | null |
