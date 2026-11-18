"""Latest-response memory used by ZSS7."""

from dataclasses import dataclass


def format_frame_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


@dataclass
class ResponseMemory:
    content: str = ""
    time: str = ""

    def update(self, accepted):
        if accepted:
            self.content = accepted[-1]["content"]
            self.time = format_frame_time(accepted[-1]["time"])

    def render(self):
        return f"Last response at time {self.time}: {self.content}"
