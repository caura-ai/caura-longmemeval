from longmemeval.dataset import LongMemEvalDataset

ds = LongMemEvalDataset()
items_map = {it.question_id: it for it in ds.load_items()}
item = items_map["3a704032"]

print("Question:", item.question)
print("Gold:", item.answer)
print("Question date:", item.question_date)

for s_idx, session in enumerate(item.haystack_sessions):
    sid = item.haystack_session_ids[s_idx]
    sdate = item.haystack_dates[s_idx]
    for t in session:
        if any(w in t['content'].lower() for w in ['plant', 'lily', 'snake', 'succulent', 'fern', 'pothos', 'monstera']):
            if t['role'] == 'user':
                print(f"Session {sid} ({sdate}): {t['content']}\n")
