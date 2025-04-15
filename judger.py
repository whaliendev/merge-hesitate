from enum import Enum
from typing import List
import re
from model import ConflictBlockModel


class BaseJudger:
    """
    Base class for conflict resolution classification.
    Provides common functionality for analyzing conflict resolutions.
    """

    def __init__(self, cb: ConflictBlockModel):
        self._our_lines = cb.ours
        self._base_lines = cb.base
        self._their_lines = cb.theirs
        self._merged_lines = cb.merged
        # Deflated versions remove all whitespace for content comparison
        self._deflated_ours = BaseJudger.deflateCodeList(cb.ours)
        self._deflated_base = BaseJudger.deflateCodeList(cb.base)
        self._deflated_theirs = BaseJudger.deflateCodeList(cb.theirs)
        self._deflated_merged = BaseJudger.deflateCodeList(cb.merged)

    def judge(self) -> List[str]:
        """Abstract method to be implemented by specific judgers"""
        pass

    @staticmethod
    def deflateCodeList(code: List[str]) -> str:
        """Remove all whitespace from code for content comparison"""
        joined = "\n".join(code)
        return re.sub(r"\s", "", joined)


class ClassifierJudger(BaseJudger):
    """
    Classifies conflict resolutions based on the relationship between
    the original versions and the merged result.

    Classification Categories:
    - OURS: Resolved by taking our version
    - THEIRS: Resolved by taking their version
    - BASE: Resolved by keeping base version
    - CONCAT: Resolved by concatenating two versions
    - DELETION: Resolved by deleting all content
    - INTERLEAVE: Resolved by mixing lines from different versions
    - NEWCODE: Resolved by adding new content not in any version
    - UNRESOLVED: Conflict markers still present
    - UNKNOWN: Cannot be classified into above categories
    """

    class Label(Enum):
        OURS = "ours"
        BASE = "base"
        THEIRS = "theirs"
        CONCAT = "concat"
        DELETION = "deletion"
        INTERLEAVE = "interleave"  # hard to auto merge
        NEWCODE = "newcode"  # almost impossible
        UNRESOLVED = "unresolved"
        UNKNOWN = "unknown"

    def judge(self) -> List[str]:
        if self.is_unresolved():
            return [ClassifierJudger.Label.UNRESOLVED.value]
        if self.accept_ours():
            return [ClassifierJudger.Label.OURS.value]
        if self.accept_theirs():
            return [ClassifierJudger.Label.THEIRS.value]
        if self.accept_base():
            return [ClassifierJudger.Label.BASE.value]
        if self.delete_all_revisions():
            return [ClassifierJudger.Label.DELETION.value]
        if self.concat_of_two_revisions():
            return [ClassifierJudger.Label.CONCAT.value]
        if self.interleave_of_revisions():
            return [ClassifierJudger.Label.INTERLEAVE.value]
        if self.new_code_introduced():
            return [ClassifierJudger.Label.NEWCODE.value]
        return [ClassifierJudger.Label.UNKNOWN.value]

    def is_unresolved(self) -> bool:
        has_conflict_line = (
            lambda x: x.startswith("<<<<<<<")
            or x.startswith("|||||||")
            or x.startswith("=======")
            or x.startswith(">>>>>>>")
        )
        return any(map(has_conflict_line, self._merged_lines))

    def accept_ours(self) -> bool:
        return self._deflated_ours == self._deflated_merged

    def accept_theirs(self) -> bool:
        return self._deflated_theirs == self._deflated_merged

    def accept_base(self) -> bool:
        return self._deflated_base == self._deflated_merged

    def concat_of_two_revisions(self) -> bool:
        # any side is empty, return false
        if not self._deflated_ours or not self._deflated_theirs:
            return False

        def concat_length_not_equal():
            return (
                (
                    len(self._deflated_ours) + len(self._deflated_theirs)
                    != len(self._deflated_merged)
                )
                and (
                    len(self._deflated_ours) + len(self._deflated_base)
                    != len(self._deflated_merged)
                )
                and (
                    len(self._deflated_theirs) + len(self._deflated_base)
                    != len(self._deflated_merged)
                )
            )

        if concat_length_not_equal():
            return False

        return (
            self._deflated_ours + self._deflated_theirs == self._deflated_merged
            or self._deflated_ours + self._deflated_base == self._deflated_merged
            or self._deflated_theirs + self._deflated_base == self._deflated_merged
            or self._deflated_theirs + self._deflated_ours == self._deflated_merged
            or self._deflated_base + self._deflated_ours == self._deflated_merged
            or self._deflated_base + self._deflated_theirs == self._deflated_merged
        )

    def new_code_introduced(self) -> bool:
        complete_set = set(self._our_lines + self._base_lines + self._their_lines)
        return any(map(lambda x: x not in complete_set, self._merged_lines))

    def delete_all_revisions(self) -> bool:
        return (
            self._deflated_ours != ""
            or self._deflated_base != ""
            or self._deflated_theirs != ""
        ) and self._deflated_merged == ""

    def interleave_of_revisions(self) -> bool:
        complete_set = set(self._our_lines + self._base_lines + self._their_lines)
        return all(map(lambda x: x in complete_set, self._merged_lines))


class MergebotJudger(BaseJudger):
    """
    Classifies conflicts based on their characteristics and complexity.

    Classification Categories:
    - STYLE_RELATED: Conflicts that only differ in code style
    - BASE_UNDERUTILIZED: One side matches the base version
    - COMPLEX_CONFLICT: Complex conflicts requiring manual resolution
    - ONE_SIDE_DELETION: One side deleted the content
    - BASE_EMPTY: No content in base version
    """

    class Label(Enum):
        STYLE_RELATED = "style_related"
        BASE_UNDERUTILIZED = "base_underutilized"
        COMPLEX_CONFLICT = "complex_conflict"
        ONE_SIDE_DELETION = "one_side_deletion"
        BASE_EMPTY = "base_empty"

    def judge(self) -> List[str]:
        if self.is_style_related():
            return [MergebotJudger.Label.STYLE_RELATED.value]
        if self.is_base_underutilized():
            return [MergebotJudger.Label.BASE_UNDERUTILIZED.value]
        if self.is_one_side_deletion():
            return [MergebotJudger.Label.ONE_SIDE_DELETION.value]
        if self.is_base_empty():
            return [MergebotJudger.Label.BASE_EMPTY.value]
        return [MergebotJudger.Label.COMPLEX_CONFLICT.value]

    def is_style_related(self) -> bool:
        return self._deflated_ours == self._deflated_theirs

    def is_base_underutilized(self) -> bool:
        return (
            self._deflated_base == self._deflated_ours
            or self._deflated_base == self._deflated_theirs
        )

    def is_one_side_deletion(self) -> bool:
        return self._deflated_ours == "" or self._deflated_theirs == ""

    def is_base_empty(self) -> bool:
        return self._deflated_base == ""
