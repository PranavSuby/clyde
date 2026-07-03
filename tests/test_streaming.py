from clyde.agent import StreamStyler, ThinkFilter


def collect(filter_obj, chunks):
    events = []
    for c in chunks:
        events += filter_obj.feed(c)
    events += filter_obj.flush()
    return events


def test_thinkfilter_split_tags_across_chunks():
    f = ThinkFilter()
    ev = collect(f, ["<thi", "nk>reason", "ing</th", "ink>ans", "wer"])
    assert "".join(t for k, t in ev if k == "thinking") == "reasoning"
    assert "".join(t for k, t in ev if k == "text") == "answer"


def test_thinkfilter_no_tags():
    f = ThinkFilter()
    ev = collect(f, ["hello ", "world"])
    assert "".join(t for k, t in ev if k == "text") == "hello world"
    assert not [t for k, t in ev if k == "thinking"]


def test_thinkfilter_flush_mid_think():
    f = ThinkFilter()
    ev = collect(f, ["<think>never closed"])
    assert "".join(t for k, t in ev if k == "thinking") == "never closed"


def test_styler_fences_and_headers():
    s = StreamStyler()
    text = "intro\n```python\ncode line\n```\nafter\n# Header\n"
    segments = []
    for ch in [text[i:i + 4] for i in range(0, len(text), 4)]:
        segments += s.feed(ch)
    segments += s.flush()
    joined = "".join(seg for seg, _ in segments)
    assert joined == text  # styling must never alter content
    # map each character to the style it was printed with
    char_styles = []
    for seg, st in segments:
        char_styles += [st] * len(seg)
    start = text.index("code line")
    assert set(char_styles[start:start + len("code line")]) == {"cyan"}
    start = text.index("# Header")
    assert set(char_styles[start:start + len("# Header")]) == {"bold"}
    start = text.index("after")
    assert set(char_styles[start:start + len("after")]) == {None}


def test_styler_preserves_exact_content_random_chunks():
    s = StreamStyler()
    text = "a\n```\nx = 1\n\n```\ntail without newline"
    out = []
    for ch in [text[i:i + 3] for i in range(0, len(text), 3)]:
        out += s.feed(ch)
    out += s.flush()
    assert "".join(seg for seg, _ in out) == text
