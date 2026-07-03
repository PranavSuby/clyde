"""Dependency-free streaming text utilities, shared with Clyde Desktop."""


class ThinkFilter:
    """Split a token stream into ('text', s) / ('thinking', s) on <think> tags.

    Handles tags split across chunk boundaries by holding back any suffix
    that could be the start of a tag.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, s: str) -> list[tuple[str, str]]:
        self.buf += s
        out = []
        while True:
            tag = self.CLOSE if self.in_think else self.OPEN
            kind = "thinking" if self.in_think else "text"
            idx = self.buf.find(tag)
            if idx >= 0:
                if idx > 0:
                    out.append((kind, self.buf[:idx]))
                self.buf = self.buf[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            # hold back the longest suffix that could start the tag
            keep = 0
            for i in range(1, len(tag)):
                if self.buf.endswith(tag[:i]):
                    keep = i
            emit = self.buf[: len(self.buf) - keep] if keep else self.buf
            self.buf = self.buf[len(self.buf) - keep:] if keep else ""
            if emit:
                out.append((kind, emit))
            return out

    def flush(self) -> list[tuple[str, str]]:
        kind = "thinking" if self.in_think else "text"
        out = [(kind, self.buf)] if self.buf else []
        self.buf = ""
        return out
