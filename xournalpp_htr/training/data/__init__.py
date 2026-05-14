import datetime
import uuid
from dataclasses import dataclass

from xournalpp_htr.documents import Stroke


@dataclass
class BBox:
    text: str
    point_1_x: float
    point_1_y: float
    point_2_x: float
    point_2_y: float
    capture_date: datetime.datetime
    uuid: str
    rect_reference: int | None
    strokes: list[Stroke] | None

    def __str__(self) -> str:
        return str(self.capture_date)

    @staticmethod
    def get_new_uuid() -> str:
        return str(uuid.uuid4())
