"""Table 2, v1 column: sense v1's own reading of the table's reactions under three requests.

v1 had no labelling guide, so its column comes from the model: v1's prompt and reading rule (the first answer
token decides; null under 0.70 confidence), with the v1 adapter from the level-3 archive:
  uv run --with torch --with transformers --with peft python src/sense_v1_table.py <archive>/sense/sense1-adapter
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
ADAPTER = sys.argv[1]
LABELS = ["0", "20", "80", "100", "null"]
SYSTEM = ("You read how a user reacted to an AI assistant and score the assistant's attempt. Answer with one of: "
          "100 (exactly right), 80 (acceptable), 20 (needs adjusting), 0 (wrong), null (unclear).")
REQS = ["I need black running shoes in size 9, under 60 dollars.", "Write a short birthday message for my sister.", "Book a table for two at an Italian restaurant tonight at 7."]
ROWS = [("Perfect, exactly what I needed", "replied", "Perfect, exactly what I needed!"),
        ("Ok, that works", "replied", "Ok, that works."),
        ("Great, now can you also...", "replied", "Great, now can you also find me some running socks?"),
        ("next request, no comment", "new request", "Find me a waterproof jacket for hiking."),
        ("Thank you so much for trying!", "replied", "Thank you so much for trying! I'll look on my own."),
        ("I guess that'll do", "replied", "I guess that'll do."),
        ("hmm", "replied", "hmm"),
        ("silence", "said nothing", ""),
        ("I said size 9", "asked to adjust", "I said size 9."),
        ("No, that's not what I asked", "replied", "No, that's not what I asked."),
        ("repeats the request", "repeated the request", None)]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32), ADAPTER).to(dev).eval()
ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LABELS]
for REQ in REQS:
  print("==", REQ)
  for name, move, said in ROWS:
      msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(
          [f"Request: {REQ}", f"The user's next move: {move}", f"What the user said: {(said if said is not None else REQ) or '(nothing)'}", "Score:"])}]
      text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
      with torch.no_grad():
          logits = model(**tok(text, return_tensors="pt").to(dev)).logits[0, -1, ids].float()
      p = torch.softmax(logits, 0).tolist()
      b = max(range(5), key=p.__getitem__)
      print(f"{name:34s} {LABELS[b] if p[b] >= 0.7 else 'null':5s} best={LABELS[b]} p={p[b]:.2f}  all=" + " ".join(f"{l}:{x:.2f}" for l, x in zip(LABELS, p)))
