"""E15 check: do the notes reach the model? For each model, builds the first prompt of a practice task through the model's
own BFCL handler (as notes.py does, but with no server: only the tokenizer) with a marker note, and checks the marker
is in the text the model would be sent.

    python -m lab.e15.check_notes <bfcl model> [<local weights folder>]
"""
import json
import sys

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.utils import populate_test_cases_with_predefined_functions
from transformers import AutoTokenizer

from lab.e15.notes import NOTES_HEADER

MARK = "- E15-MARKER: check the state before acting."


def main():
    registry, path = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if registry.startswith("e14-"):
        from lab.e14.new_handlers import register
        register()
    cfg = MODEL_CONFIG_MAPPING[registry]
    h = cfg.model_handler(model_name=cfg.model_name, temperature=0.001, registry_name=registry, is_fc_model=cfg.is_fc_model)
    h.model_path_or_id = path or cfg.model_name
    h.tokenizer = AutoTokenizer.from_pretrained(h.model_path_or_id, trust_remote_code=True)
    def prompt_for(notes: bool) -> str:  # as bfcl's inference does: the first turn is changed in place
        q = {"id": "multi_turn_base_0", "question": [[{"role": "user", "content": "Start the car, please."}]],
             "initial_config": {}, "involved_classes": ["VehicleControlAPI"]}
        populate_test_cases_with_predefined_functions([q])
        if notes:
            q["question"][0].insert(0, {"role": "system", "content": NOTES_HEADER + "\n" + MARK})
        data = h._pre_query_processing_prompting(q)
        data = h.add_first_turn_message_prompting(data, q["question"][0])
        return h._format_prompt(data["message"], data["function"])

    prompt, plain = prompt_for(True), prompt_for(False)
    ok = MARK in prompt and NOTES_HEADER in prompt
    # the notes must add to the prompt, not replace anything: every tool name in the plain prompt is still there
    kept = all(t in prompt for t in ("startEngine", "lockDoors")) == all(t in plain for t in ("startEngine", "lockDoors"))
    i = prompt.find(MARK)
    print(json.dumps({"registry": registry, "notes_in_prompt": ok, "tools_in_plain": "startEngine" in plain,
                      "tools_kept": kept, "chars_plain": len(plain), "chars_with_notes": len(prompt),
                      "around": prompt[max(0, i - 160): i + len(MARK) + 40] if i >= 0 else prompt[:300]}))


if __name__ == "__main__":
    main()
