# Labelling guide: how a user reacted

The feedback sense reads how a person reacted to something an AI assistant did for them, and scores the
assistant's attempt. It reads the person, never the work: it does not judge whether the answer was correct,
only what the person's next move and words show about how they feel about it. It must work for any task.

## What you see

- **Request:** what the person asked for (sometimes missing).
- **Next move:** what kind of turn came next: replied, new request, asked to adjust, repeated the request,
  left the session, said nothing.
- **Words:** what the person said next (may be empty).

## The five scores

Two clear groups with a gap between them. When a reaction does not clearly belong to one group, it is unclear.

| Score | Meaning | Typical behaviour | Typical words |
| --- | --- | --- | --- |
| 100 | Exactly right | Explicit praise or thanks that show real satisfaction | "Perfect, exactly what I needed", "Awesome, thank you!", "That's the one" |
| 80 | Acceptable, or good enough to build on | Moves on to a new, different request without complaint; a plain "ok, thanks"; keeps going by building on the result, including new preferences the request did not mention | "Ok thanks", "Great, now can you also...", "Let's also add...", "Make it shorter", "Same but in blue", the next task with no comment |
| null | Unclear | The reaction does not show whether it worked | Silence, "hmm", "I guess", "ok..." in a flat tone, off-topic, leaving the session, settling for it without saying whether it is good |
| 20 | Needs adjusting | One part of the result missed what was asked; most of it was fine | "I wanted it in blue", "I said size 9", "I asked for under 100 words", "That's over the budget I gave" |
| 0 | Wrong | Rejects it, repeats or rephrases the same request, points out it did not do what was asked, gives up, gets angry | "No, that's not what I asked", "I said size 9", "That's not my style at all", "Forget it", "wtf" |

## Hard cases

**Continuing the conversation is not failure.** When the person keeps going after a result ("do this next",
"now make it...", "let's also...", building on it, or changing their mind after a useful answer), the assistant
did well enough to let them think further. Score **80**.

**Building on it vs correcting it.** The line is whether the result missed what was asked, not whether the
person asks for a change:

- A new preference or refinement the request did not mention ("make it shorter", "use camelCase", "same but in
  blue", "now make it formal") is building on the result: **80**.
- Only when the words show the result missed the request ("I said...", "I asked for...", "that's not what I
  meant", "like I asked") is it **20** (one detail off) or **0** (the whole thing missed).
- When the words do not say whether it was asked before, and the request is not shown: **null**.

| Reaction | Did the request ask for it? | Score |
| --- | --- | --- |
| "Can you rename the variables to camelCase?" | No: a new preference | 80 |
| "Use camelCase like I asked" | Yes: it was missed | 20 |
| "Make it shorter" | No length asked | 80 |
| "I asked for under 100 words" | Yes | 20 |
| "Same but in blue" | Blue not asked before | 80 |
| "I wanted it in blue" | Blue was asked | 20 |

**Polite unhappiness.** "Thanks!" followed by asking the same thing again is not praise: score by the
behaviour (**0**). Thanks with no further sign either way is **80** at most, not 100.

**Moved on vs gave up.** A new, different request without complaint is acceptance (**80**). Leaving the session
or switching away after a complaint is giving up (**0**). Leaving with no words at all is **null**.

**Settling.** "I guess that'll do", "whatever, fine": the person accepts reluctantly and the reaction does not
show the result was good. Score **null**.

**"Not my style", "not really my thing".** A rejection of taste: **0** when the whole result is rejected,
**20** when they ask for something similar but different that they had described.

**Task-specific words you do not understand** ("segfault", "runs small", "it bounced"): if the meaning is not
clear from the words and behaviour, score **null** rather than guess.

**Silence.** Nothing said and no next move: **null**.

## Examples across tasks

| Task | Next move | Words | Score |
| --- | --- | --- | --- |
| Coding | replied | "It works now, thanks!" | 100 |
| Coding | replied | "Still getting the same error" | 0 |
| Coding | asked to adjust | "Can you rename the variables to camelCase?" | 80 |
| Coding | asked to adjust | "I said camelCase" | 20 |
| Writing | replied | "Love it. Now a shorter version for Twitter?" | 80 |
| Writing | replied | "Too formal, I said casual" | 20 |
| Booking | new request | "Now I need a taxi to the airport" | 80 |
| Booking | repeated the request | "I need a hotel in the centre, with parking" | 0 |
| Recommending | replied | "Seen it. Not really my thing" | 0 |
| Recommending | replied | "Ooh, haven't seen that one, adding it" | 100 |
| Search | replied | "hmm" | null |
| Support | left the session | (nothing) | null |
| Support | replied | "Thanks!" then asks the same question again | 0 |
| Any | said nothing | (nothing) | null |
| Any | replied | "I guess that's fine" | null |
