import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) >= 3:
        p_before = Path(sys.argv[1])
        p_after = Path(sys.argv[2])
    else:
        p_before = Path("outputs/caura-50-adaptive-v4/results.json")
        p_after = Path("outputs/caura-50-agentic-v1/results.json")

    if p_before.is_dir():
        p_before = p_before / "results.json"
    if p_after.is_dir():
        p_after = p_after / "results.json"

    with open(p_before, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(p_after, "r", encoding="utf-8") as f:
        after = json.load(f)

    b_sum = before["summary"]
    a_sum = after["summary"]

    print("=" * 75)
    print(f"{'METRIC':<30} | {'BEFORE (direct)':<18} | {'AFTER (agentic-v1)':<18}")
    print("-" * 75)
    print(f"{'Total Questions':<30} | {b_sum['total_questions']:<18} | {a_sum['total_questions']:<18}")
    print(f"{'Correct Questions':<30} | {b_sum['correct_questions']:<18} | {a_sum['correct_questions']:<18}")
    print(f"{'Overall Accuracy':<30} | {b_sum['overall_accuracy']*100:<17.1f}% | {a_sum['overall_accuracy']*100:<17.1f}%")
    print(f"{'Projected Official Accuracy':<30} | {b_sum.get('projected_official_accuracy', 0)*100:<17.1f}% | {a_sum.get('projected_official_accuracy', 0)*100:<17.1f}%")
    print("=" * 75)
    print("CATEGORY BREAKDOWN:")
    print("-" * 75)

    categories = list(b_sum["by_question_type"].keys())
    for cat in categories:
        b_acc = b_sum["by_question_type"].get(cat, {}).get("accuracy", 0.0) * 100
        b_cor = b_sum["by_question_type"].get(cat, {}).get("correct", 0)
        a_acc = a_sum["by_question_type"].get(cat, {}).get("accuracy", 0.0) * 100
        a_cor = a_sum["by_question_type"].get(cat, {}).get("correct", 0)
        tot = b_sum["by_question_type"].get(cat, {}).get("total", 0)
        diff = a_acc - b_acc
        sign = "+" if diff > 0 else ""
        print(f"{cat:<28} | {b_cor}/{tot} ({b_acc:>5.1f}%)       | {a_cor}/{tot} ({a_acc:>5.1f}%) [{sign}{diff:+.1f}%]")
    print("=" * 75)

    # Let's inspect which specific questions changed
    b_list = before.get("questions") or before["summary"].get("results", [])
    a_list = after.get("questions") or after["summary"].get("results", [])
    b_by_id = {r["question_id"]: r for r in b_list}
    a_by_id = {r["question_id"]: r for r in a_list}

    flipped_to_pass = []
    flipped_to_fail = []

    for qid, a_res in a_by_id.items():
        b_res = b_by_id.get(qid)
        if not b_res:
            continue
        if not b_res["correct"] and a_res["correct"]:
            flipped_to_pass.append(a_res)
        elif b_res["correct"] and not a_res["correct"]:
            flipped_to_fail.append((b_res, a_res))

    print(f"\nQUESTIONS FLIPPED FROM FAIL -> PASS ({len(flipped_to_pass)}):")
    for r in flipped_to_pass:
        q_text = r['question'].encode('ascii', 'replace').decode('ascii')
        g_text = str(r['gold_answer']).encode('ascii', 'replace').decode('ascii')
        h_text = str(r['hypothesis']).encode('ascii', 'replace').decode('ascii')
        print(f"  + #{r['question_id']} ({r['question_type']}): {q_text}")
        print(f"    Gold: {g_text}")
        print(f"    Hypothesis: {h_text}")
        print()

    print(f"QUESTIONS FLIPPED FROM PASS -> FAIL ({len(flipped_to_fail)}):")
    for b_res, a_res in flipped_to_fail:
        q_text = a_res['question'].encode('ascii', 'replace').decode('ascii')
        g_text = str(a_res['gold_answer']).encode('ascii', 'replace').decode('ascii')
        b_text = str(b_res['hypothesis']).encode('ascii', 'replace').decode('ascii')
        a_text = str(a_res['hypothesis']).encode('ascii', 'replace').decode('ascii')
        print(f"  - #{a_res['question_id']} ({a_res['question_type']}): {q_text}")
        print(f"    Gold: {g_text}")
        print(f"    Before: {b_text}")
        print(f"    After:  {a_text}")
        print()

if __name__ == "__main__":
    main()
