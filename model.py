from dataclasses import dataclass
from typing import List


@dataclass
class ConflictBlockModel:
    """
    Internal model for processing conflict blocks.
    Contains the raw content and position information of a conflict block.

    Attributes:
        index: Sequential number of this conflict block in the file
        ours: Our version of the changes (as list of lines)
        base: Base version of the code (as list of lines)
        theirs: Their version of the changes (as list of lines)
        merged: The resolved version (as list of lines)
        labels: Classification labels for this conflict
        start_line: Starting line number of the conflict block
        end_line: Ending line number of the conflict block
        resolved_start_line: Starting line number in the resolved content
        resolved_end_line: Ending line number in the resolved content
    """

    index: int
    ours: List[str]
    base: List[str]
    theirs: List[str]
    merged: List[str]
    labels: List[str]
    start_line: int
    end_line: int
    resolved_start_line: int
    resolved_end_line: int


@dataclass
class ConflictBlock:
    """
    Public interface for conflict blocks.
    Contains the processed and formatted content of a conflict block.

    Attributes:
        index: Sequential number of this conflict block in the file
        ours: Our version of the changes (as single string)
        theirs: Their version of the changes (as single string)
        base: Base version of the code (as single string)
        merged: The resolved version (as single string)
        labels: Classification labels for this conflict
        start_line: Starting line number of the conflict block
        end_line: Ending line number of the conflict block
        resolved_start_line: Starting line number in the resolved content
        resolved_end_line: Ending line number in the resolved content
    """

    index: int
    ours: str
    theirs: str
    base: str
    merged: str
    labels: List[str]
    start_line: int
    end_line: int
    resolved_start_line: int
    resolved_end_line: int
