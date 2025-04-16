from typing import List
from model import ConflictBlock, ConflictBlockModel
from judger import ClassifierJudger, MergebotJudger
from string_op import string_similarity
import difflib

# Conflict markers in git merge conflicts
OUR_CONFLICT_MARKER = "<<<<<<<"  # Marks the start of our changes
BASE_CONFLICT_MARKER = "|||||||"  # Marks the separation between base and their changes
THEIR_CONFLICT_MARKER = "======="  # Marks their changes
END_CONFLICT_MARKER = ">>>>>>>"  # Marks the end of conflict block


def _get_code_snippet(lines: List[str], start: int, end: int) -> List[str]:
    """get slice of (start, end) from lines, start and end are exclusive, index starts from 0"""
    return lines[start + 1 : end]


def _align_line_scan(
    anchor: List[str],
    merged: List[str],
    is_suffix: bool,
    start_index: int,
    merged_file: str,
) -> int:
    """
    Align and find the best matching position between anchor lines and merged content.

    Algorithm:
    1. For prefix matching (is_suffix=False): Scan backwards from start_index
    2. For suffix matching (is_suffix=True): Scan forwards from start_index
    3. Uses a sliding window approach to find the best match
    4. Prefers matches that are closer to the start_index

    Args:
        anchor: The lines to match against (prefix or suffix of conflict block)
        merged: The merged content to search in
        is_suffix: Whether we're matching a suffix (True) or prefix (False)
        start_index: The starting position in merged content

    Returns:
        The best matching position in the merged content
    """
    pivot = []
    source = []

    if not is_suffix:
        pivot.extend(reversed(anchor))
        source.extend(reversed(merged[start_index:]))
    else:
        pivot.extend(anchor)
        source.extend(merged[start_index + 1 :])

    if len(pivot) == 0:
        if is_suffix:
            return len(merged)
        else:
            return -1

    max_align = -1
    loc = 0
    best_similarity = -1.0  # 最佳相似度

    for i in range(len(source)):
        if max_align > 10:  # Stop if we found a good enough match
            break

        if pivot[0] == source[i]:
            j, k = i, 0
            while k < len(pivot) and j < len(source) and source[j] == pivot[k]:
                k += 1
                j += 1

            # For prefix matching, prefer matches closer to the end
            # For suffix matching, take the longest match
            if k > max_align:
                max_align = k
                loc = i

                if k > 10:
                    break  # stop if we found a good enough match
            elif k == max_align and k > 0:
                # use a large context window to prevent the interruption of pivot with a close merge conflict
                context_size = 40
                # current_context = []
                # if not is_suffix:
                current_context = (
                    source[i : i + k + context_size]
                    if i + k + context_size <= len(source)
                    else source[i : len(source)]
                )
                # else:
                #     current_context = source[max(0, i - context_size) : i + k]

                # prev_context = []
                # if not is_suffix:
                prev_context = (
                    source[loc : loc + k + context_size]
                    if loc + k + context_size <= len(source)
                    else source[loc : len(source)]
                )
                # else:
                #     prev_context = source[max(0, loc - context_size) : loc + k]

                current_text = "\n".join(current_context)
                prev_text = "\n".join(prev_context)

                # calculate string similarity
                current_similarity = string_similarity(
                    current_text, "\n".join(pivot[: min(len(pivot), k + context_size)])
                )
                prev_similarity = string_similarity(
                    prev_text, "\n".join(pivot[: min(len(pivot), k + context_size)])
                )

                # if current match has higher similarity, update the best match
                if (
                    current_similarity > prev_similarity
                    and current_similarity > best_similarity
                ):
                    best_similarity = current_similarity
                    loc = i
    if best_similarity != -1 and best_similarity < 0.5:
        print(f"best_similarity: {best_similarity}, merged_file: {merged_file}")

    return len(merged) - loc + 1 if not is_suffix else start_index + loc + 1


def process_conflict_blocks_v1_old(
    conflict_content: str, truth_content: str, merged_file: str
) -> List[ConflictBlock]:
    """
    Process git merge conflict content and extract conflict blocks with their resolutions.

    Algorithm:
    1. Parse the conflict content to identify conflict blocks
    2. For each conflict block:
       - Extract our changes, base version, and their changes
       - Find the corresponding resolved content in the merged version
       - Apply conflict resolution classification

    Args:
        conflict_content: The content with conflict markers
        truth_content: The resolved content after conflict resolution

    Returns:
        List of ConflictBlock objects containing the parsed and classified conflicts
    """
    conflict_block_models: List[ConflictBlockModel] = []
    index_in_file = 0
    conflict_lines = conflict_content.splitlines()
    conflict_line_cnt = len(conflict_lines)

    for index, line in enumerate(conflict_lines):
        if line.startswith(OUR_CONFLICT_MARKER):
            cb = ConflictBlockModel(index_in_file, [], [], [], [], [], 0, 0, 0, 0)
            index_in_file += 1
            j = index
            k = index

            cb.start_line = index + 1  # line number starts from 1

            while j + 1 < conflict_line_cnt and not conflict_lines[j + 1].startswith(
                BASE_CONFLICT_MARKER
            ):
                j += 1
            j += 1  # j now points to the line with base conflict marker
            cb.ours = _get_code_snippet(conflict_lines, k, j)
            k = j

            while j + 1 < conflict_line_cnt and not conflict_lines[j + 1].startswith(
                THEIR_CONFLICT_MARKER
            ):
                j += 1
            j += 1
            cb.base = _get_code_snippet(conflict_lines, k, j)
            k = j

            while j + 1 < conflict_line_cnt and not conflict_lines[j + 1].startswith(
                END_CONFLICT_MARKER
            ):
                j += 1
            j += 1
            cb.theirs = _get_code_snippet(conflict_lines, k, j)
            k = j
            cb.end_line = j + 1  # line number
            conflict_block_models.append(cb)

    truth_lines = truth_content.splitlines()
    start_index = 0
    for i, cb in enumerate(conflict_block_models):
        if i == 0:
            prefix = _get_code_snippet(conflict_lines, -1, cb.start_line - 1)
        else:
            previous_conflict_block = conflict_block_models[i - 1]
            interval = cb.start_line - previous_conflict_block.end_line - 1
            prefix = _get_code_snippet(
                truth_lines,
                previous_conflict_block.resolved_start_line,
                previous_conflict_block.resolved_end_line + interval,
            )
        suffix = _get_code_snippet(conflict_lines, cb.end_line - 1, conflict_line_cnt)
        start_line = _align_line_scan(
            prefix, truth_lines, False, start_index, merged_file
        )
        # print(f"start_line: {start_line}")
        cb.resolved_start_line = start_line
        start_index = start_line
        end_line = _align_line_scan(suffix, truth_lines, True, start_index, merged_file)
        # print(f"end_line: {end_line}")
        cb.resolved_end_line = end_line
        start_index = end_line
        cb.merged = _get_code_snippet(truth_lines, start_line - 2, end_line)
        # judge strategy
        jugers = [ClassifierJudger(cb), MergebotJudger(cb)]
        for judger in jugers:
            labels = judger.judge()
            cb.labels.extend(labels)

    conflict_blocks = [
        ConflictBlock(
            model.index,
            "\n".join(model.ours),
            "\n".join(model.theirs),
            "\n".join(model.base),
            "\n".join(model.merged),
            model.labels,
            model.start_line,
            model.end_line,
            model.resolved_start_line,
            model.resolved_end_line,
        )
        for model in conflict_block_models
    ]

    return conflict_blocks


def process_conflict_blocks_v2(
    conflict_content: str, truth_content: str, merged_file: str
) -> List[ConflictBlock]:
    """
    Process git merge conflict content and extract conflict blocks with their resolutions.
    Version 5 (Revised V2): Uses difflib, full context, two-pass approach with
    independent context matching (Pass 1) and sequential boundary resolution
    with enforced suffix re-scan (after start line) when end < start is detected.

    Args:
        conflict_content: The content with conflict markers
        truth_content: The resolved content after conflict resolution
        merged_file: The path to the merged file (used for logging/debugging)

    Returns:
        List of ConflictBlock objects containing the parsed and classified conflicts
    """
    conflict_block_models: List[ConflictBlockModel] = []
    index_in_file = 0
    conflict_lines = conflict_content.splitlines()
    conflict_line_cnt = len(conflict_lines)

    # --- Step 1: Parse Conflict Blocks ---
    i = 0
    while i < conflict_line_cnt:
        if conflict_lines[i].startswith(OUR_CONFLICT_MARKER):
            cb = ConflictBlockModel(
                index=index_in_file,
                ours=[],
                theirs=[],
                base=[],
                merged=[],
                labels=[],
                start_line=0,
                end_line=0,
                resolved_start_line=-1,
                resolved_end_line=-1,
            )
            cb.prefix_context = []
            cb.suffix_context = []
            cb.best_prefix_score = -1.0
            cb.best_prefix_match_end = -1
            cb.best_suffix_score = -1.0
            cb.best_suffix_match_start = -1
            index_in_file += 1
            cb.start_line = i + 1
            start_ours = i + 1
            i += 1
            while i < conflict_line_cnt and not conflict_lines[i].startswith(
                BASE_CONFLICT_MARKER
            ):
                i += 1
            if i >= conflict_line_cnt:
                raise ValueError(
                    f"Unterminated OURS block starting at line {cb.start_line}"
                )
            cb.ours = conflict_lines[start_ours:i]
            start_base = i + 1
            i += 1
            while i < conflict_line_cnt and not conflict_lines[i].startswith(
                THEIR_CONFLICT_MARKER
            ):
                i += 1
            if i >= conflict_line_cnt:
                raise ValueError(
                    f"Unterminated BASE block starting at line {cb.start_line}"
                )
            cb.base = conflict_lines[start_base:i]
            start_theirs = i + 1
            i += 1
            while i < conflict_line_cnt and not conflict_lines[i].startswith(
                END_CONFLICT_MARKER
            ):
                i += 1
            if i >= conflict_line_cnt:
                raise ValueError(
                    f"Unterminated THEIRS block starting at line {cb.start_line}"
                )
            cb.theirs = conflict_lines[start_theirs:i]
            cb.end_line = i + 1
            conflict_block_models.append(cb)
        i += 1

    if not conflict_block_models:
        return []

    truth_lines = truth_content.splitlines()
    num_truth_lines = len(truth_lines)

    # --- Step 2: Define Similarity Calculation ---
    def calculate_similarity(lines1: List[str], lines2: List[str]) -> float:
        if not lines1 and not lines2:
            return 1.0
        if not lines1 or not lines2:
            return 0.0
        return difflib.SequenceMatcher(None, lines1, lines2).ratio()

    # --- Step 3: Identify Full Context for Each Block ---
    for idx, cb in enumerate(conflict_block_models):
        prefix_start = 0
        if idx > 0:
            prefix_start = conflict_block_models[idx - 1].end_line
        cb.prefix_context = conflict_lines[prefix_start : cb.start_line - 1]
        suffix_end = conflict_line_cnt
        if idx < len(conflict_block_models) - 1:
            suffix_end = conflict_block_models[idx + 1].start_line - 1
        cb.suffix_context = conflict_lines[cb.end_line : suffix_end]

    # --- Step 4: Pass 1 - Independent Best Match Search ---
    print(
        f"Starting Pass 1: Independent Context Matching for {len(conflict_block_models)} blocks in {merged_file}"
    )
    for cb in conflict_block_models:
        prefix_len = len(cb.prefix_context)
        if prefix_len == 0:
            cb.best_prefix_score = 1.0
            cb.best_prefix_match_end = 0
        else:
            best_score = -1.0
            best_index = -1
            for i in range(num_truth_lines - prefix_len + 1):
                score = calculate_similarity(
                    cb.prefix_context, truth_lines[i : i + prefix_len]
                )
                if score > best_score:
                    best_score = score
                    best_index = i + prefix_len
            cb.best_prefix_score = best_score
            cb.best_prefix_match_end = best_index if best_index != -1 else 0

        suffix_len = len(cb.suffix_context)
        if suffix_len == 0:
            cb.best_suffix_score = 1.0
            cb.best_suffix_match_start = num_truth_lines
        else:
            best_score = -1.0
            best_index = -1
            for i in range(num_truth_lines - suffix_len + 1):
                score = calculate_similarity(
                    cb.suffix_context, truth_lines[i : i + suffix_len]
                )
                if score > best_score:
                    best_score = score
                    best_index = i
            cb.best_suffix_score = best_score
            cb.best_suffix_match_start = (
                best_index if best_index != -1 else num_truth_lines
            )
        # print(f"  Block {cb.index} P1: PrefixEnd={cb.best_prefix_match_end}({cb.best_prefix_score:.2f}), SuffixStart={cb.best_suffix_match_start}({cb.best_suffix_score:.2f})")

    # --- Step 5: Pass 2 - Sequential Boundary Resolution ---
    print(f"Starting Pass 2: Sequential Boundary Resolution...")
    for idx, cb in enumerate(conflict_block_models):
        # --- Determine Resolved Start Line ---
        current_start = cb.best_prefix_match_end
        if idx > 0:
            prev_end = conflict_block_models[idx - 1].resolved_end_line
            # Enforce sequential ordering: Start cannot be before previous block's end
            current_start = max(prev_end, current_start)
        cb.resolved_start_line = max(0, current_start)

        # --- Determine Resolved End Line ---
        current_end = cb.best_suffix_match_start

        # Always check for and fix end < start condition by re-searching suffix after start
        if current_end < cb.resolved_start_line:
            print(
                f"  Warning Block {cb.index}: Initial suffix match ({current_end}) is before determined start ({cb.resolved_start_line}). Re-searching suffix AFTER start."
            )

            best_score_rescanned = -1.0
            best_index_rescanned = -1
            suffix_len = len(cb.suffix_context)
            search_start = (
                cb.resolved_start_line + 1
            )  # <-- Start search STRICTLY AFTER resolved start

            if suffix_len > 0 and search_start < num_truth_lines:
                search_end_loop = num_truth_lines - suffix_len + 1
                if search_start < search_end_loop:
                    for i in range(search_start, search_end_loop):
                        score = calculate_similarity(
                            cb.suffix_context, truth_lines[i : i + suffix_len]
                        )
                        if score > best_score_rescanned:
                            best_score_rescanned = score
                            best_index_rescanned = i
                        # Keep first best match in rescan

            if best_index_rescanned != -1:
                print(
                    f"    Rescan found suffix match at {best_index_rescanned} (Score: {best_score_rescanned:.2f}). Updating end line."
                )
                current_end = best_index_rescanned
                cb.best_suffix_score = best_score_rescanned  # Update score to reflect the actual match used
            else:
                # Fallback ONLY if rescan finds nothing
                print(
                    f"    Warning Block {cb.index}: Suffix rescan (after start) failed. Assuming empty resolution. Setting end = start ({cb.resolved_start_line})."
                )
                current_end = cb.resolved_start_line
                cb.best_suffix_score = 0.0  # Reflect failure

        cb.resolved_end_line = current_end

        # Final sanity check
        if cb.resolved_end_line < cb.resolved_start_line:
            print(
                f"  ERROR Block {cb.index}: Final end line {cb.resolved_end_line} < start line {cb.resolved_start_line}. Resetting end=start."
            )
            cb.resolved_end_line = cb.resolved_start_line

        print(
            f"  Block {cb.index}: Final match [{cb.resolved_start_line + 1}, {cb.resolved_end_line}] (Scores P:{cb.best_prefix_score:.2f}, S:{cb.best_suffix_score:.2f}) "
        )

    # --- Step 6: Finalize Merged Content & Apply Judgers ---
    conflict_blocks_final: List[ConflictBlock] = []
    for cb_model in conflict_block_models:
        start = cb_model.resolved_start_line
        end = cb_model.resolved_end_line
        if start < 0 or end < 0:
            start = max(0, start)
            end = max(0, end)
        if end < start:
            end = start
            cb_model.labels.append("MATCH_ERROR_NEGATIVE_RANGE")
        cb_model.merged = truth_lines[start:end]

        judger_start_line = start + 1
        judger_end_line = end
        cb_for_judger = ConflictBlockModel(
            cb_model.index,
            cb_model.ours,
            cb_model.theirs,
            cb_model.base,
            cb_model.merged,
            [],
            cb_model.start_line,
            cb_model.end_line,
            judger_start_line,
            judger_end_line,
        )
        current_labels = []
        try:
            jugers = [ClassifierJudger(cb_for_judger), MergebotJudger(cb_for_judger)]
            for judger in jugers:
                try:
                    current_labels.extend(judger.judge())
                except Exception as e:
                    current_labels.append(f"{type(judger).__name__}_ERROR")
            cb_model.labels.extend(current_labels)
        except NameError as ne:
            cb_model.labels.append("JUDGER_UNAVAILABLE")

        conflict_blocks_final.append(
            ConflictBlock(
                cb_model.index,
                "\n".join(cb_model.ours),
                "\n".join(cb_model.theirs),
                "\n".join(cb_model.base),
                "\n".join(cb_model.merged),
                cb_model.labels,
                cb_model.start_line,
                cb_model.end_line,
                start + 1,
                end,
            )
        )
    return conflict_blocks_final


def process_conflict_blocks(
    conflict_content: str, truth_content: str, merged_file: str
) -> List[ConflictBlock]:
    """
    Process git merge conflict content and extract conflict blocks with their resolutions.
    Version 3 (Opcode Mapping): Uses difflib.SequenceMatcher.get_opcodes() to precisely map
    conflict block boundaries (marker lines) in the conflict file to their
    corresponding line numbers in the resolved file, mimicking visual diff tools.

    Args:
        conflict_content: The content with conflict markers.
        truth_content: The resolved content after conflict resolution.
        merged_file: The path to the merged file (used for logging/debugging).

    Returns:
        List of ConflictBlock objects containing the parsed and classified conflicts.
    """
    conflict_block_models: List[ConflictBlockModel] = []
    index_in_file = 0
    conflict_lines = conflict_content.splitlines()
    conflict_line_cnt = len(conflict_lines)
    truth_lines = truth_content.splitlines()
    num_truth_lines = len(truth_lines)

    # --- Step 1: Parse Conflict Blocks & Store 0-based Original Boundaries ---
    i = 0
    while i < conflict_line_cnt:
        if conflict_lines[i].startswith(OUR_CONFLICT_MARKER):
            start_marker_orig_idx = i  # 0-based index of <<<<<<<
            cb = ConflictBlockModel(
                index=index_in_file,
                ours=[],
                theirs=[],
                base=[],
                merged=[],
                labels=[],
                start_line=start_marker_orig_idx
                + 1,  # Store 1-based marker line for final ConflictBlock
                end_line=0,  # Will be set later
                resolved_start_line=-1,  # 0-based index in truth_lines
                resolved_end_line=-1,  # 0-based index in truth_lines
            )
            # Store original 0-based indices internally
            cb.start_marker_orig_idx = start_marker_orig_idx
            cb.end_marker_orig_idx = -1

            index_in_file += 1
            start_ours = i + 1
            i += 1
            while i < conflict_line_cnt and not conflict_lines[i].startswith(
                BASE_CONFLICT_MARKER
            ):
                i += 1
            if i >= conflict_line_cnt:
                raise ValueError(
                    f"Unterminated OURS block starting at line {cb.start_line} in {merged_file}"
                )
            cb.ours = conflict_lines[start_ours:i]

            start_base = i + 1
            i += 1
            while i < conflict_line_cnt and not conflict_lines[i].startswith(
                THEIR_CONFLICT_MARKER
            ):
                i += 1
            if i >= conflict_line_cnt:
                raise ValueError(
                    f"Unterminated BASE block starting at line {cb.start_line} in {merged_file}"
                )
            cb.base = conflict_lines[start_base:i]

            start_theirs = i + 1
            i += 1
            while i < conflict_line_cnt and not conflict_lines[i].startswith(
                END_CONFLICT_MARKER
            ):
                i += 1
            if i >= conflict_line_cnt:
                raise ValueError(
                    f"Unterminated THEIRS block starting at line {cb.start_line} in {merged_file}"
                )
            cb.theirs = conflict_lines[start_theirs:i]

            cb.end_marker_orig_idx = i  # 0-based index of >>>>>>>
            cb.end_line = i + 1  # Store 1-based marker line for final ConflictBlock
            conflict_block_models.append(cb)
        i += 1

    if not conflict_block_models:
        return []

    # --- Step 2: Compute Opcodes ---
    # print(
    #     f"Starting V3 (Opcode) analysis for {len(conflict_block_models)} blocks in {merged_file}"
    # )
    # Use autojunk=False for potentially better accuracy with noisy data, though may be slower.
    matcher = difflib.SequenceMatcher(None, conflict_lines, truth_lines, autojunk=False)
    opcodes = matcher.get_opcodes()

    # --- Step 3: Build Line Mapping from Opcodes ---
    # Create a detailed mapping from conflict lines to truth lines
    # This will track each line's fate (kept, deleted, replaced)
    conflict_to_truth_map = {}  # Maps conflict line -> truth line or range

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # Direct mapping for equal lines
            for offset in range(i2 - i1):
                conflict_to_truth_map[i1 + offset] = j1 + offset
        elif tag == "delete":
            # Deleted lines map to the point where they would have been
            # Map to position in truth file where deletion happened
            for conflict_idx in range(i1, i2):
                conflict_to_truth_map[conflict_idx] = ("delete", j1)
        elif tag == "replace":
            # Replaced lines - for conflict markers, track both the start and end
            # points of the replacement in the truth file
            # Each conflict line in this range is replaced by the entire j1:j2 range
            for conflict_idx in range(i1, i2):
                conflict_to_truth_map[conflict_idx] = ("replace", j1, j2)
        elif tag == "insert":
            # Insert doesn't affect conflict line mapping directly
            # But might be relevant if we need to track inserted content
            pass

    # print(f"  Generated line mapping with {len(conflict_to_truth_map)} entries")

    # --- Step 4: Map Boundaries using Index Maps ---
    for cb in conflict_block_models:
        s_orig = cb.start_marker_orig_idx  # 0-based index of <<<<<<<
        e_orig = cb.end_marker_orig_idx  # 0-based index of >>>>>>>

        # Find resolved_start based on what happened to the start marker line
        if s_orig in conflict_to_truth_map:
            mapping = conflict_to_truth_map[s_orig]
            if isinstance(mapping, int):
                # Direct mapping (equal) - rare case where marker is kept unchanged
                # marker本身不应该包含在解决方案中，所以跳过该行
                resolved_start = mapping + 1
            elif isinstance(mapping, tuple) and mapping[0] == "delete":
                # Marker was deleted - use position where it would have been
                resolved_start = mapping[1]
            elif isinstance(mapping, tuple) and mapping[0] == "replace":
                # Marker was part of a replaced block
                # Use the start of the replacement section
                resolved_start = mapping[1]
            else:
                resolved_start = 0  # Fallback
        else:
            # print(
            #     f"  Warning: Start marker at {s_orig} not found in mapping, using fallback position 0"
            # )
            resolved_start = 0

        # Find resolved_end based on what happened to the end marker line
        if e_orig in conflict_to_truth_map:
            mapping = conflict_to_truth_map[e_orig]
            if isinstance(mapping, int):
                # Direct mapping (equal) - rare case where marker is kept unchanged
                # marker本身不应该包含在解决方案中，所以不包含该行
                resolved_end = mapping
            elif isinstance(mapping, tuple) and mapping[0] == "delete":
                # Marker was deleted - use position where it would have been
                resolved_end = mapping[1]
            elif isinstance(mapping, tuple) and mapping[0] == "replace":
                # Marker was part of a replaced block
                # Use the end of the replacement section
                resolved_end = mapping[2]
            else:
                resolved_end = num_truth_lines  # Fallback
        else:
            print(
                f"  Warning: End marker at {e_orig} not found in mapping, using fallback position {num_truth_lines}"
            )
            resolved_end = num_truth_lines

        # 处理边界情况：如果冲突块中的行在删除/替换操作中与其他行一起被处理，需要额外检查
        # 检查 s_orig-1 和 s_orig+1 的映射，帮助确定准确的边界
        if (
            s_orig > 0
            and s_orig - 1 in conflict_to_truth_map
            and s_orig + 1 in conflict_to_truth_map
        ):
            prev_mapping = conflict_to_truth_map[s_orig - 1]
            next_mapping = conflict_to_truth_map[s_orig + 1]

            # 如果前一行是直接映射，而当前标记行被删除，那么resolved_start可能需要调整
            if (
                isinstance(prev_mapping, int)
                and isinstance(mapping, tuple)
                and mapping[0] == "delete"
            ):
                resolved_start = prev_mapping + 1

            # 如果后续行有明确映射，而当前是替换，可能需要调整resolved_start
            if (
                isinstance(next_mapping, int)
                and isinstance(mapping, tuple)
                and mapping[0] == "replace"
            ):
                if next_mapping < resolved_start:
                    resolved_start = next_mapping

        # 对结束标记应用类似的逻辑
        if (
            e_orig + 1 < conflict_line_cnt
            and e_orig - 1 in conflict_to_truth_map
            and e_orig + 1 in conflict_to_truth_map
        ):
            prev_mapping = conflict_to_truth_map[e_orig - 1]
            next_mapping = conflict_to_truth_map[e_orig + 1]

            # 如果结束标记被删除，而后续行有映射，调整resolved_end
            if (
                isinstance(next_mapping, int)
                and isinstance(mapping, tuple)
                and mapping[0] == "delete"
            ):
                resolved_end = next_mapping

            # 如果前一行有明确映射，而当前是替换，可能需要调整resolved_end
            if (
                isinstance(prev_mapping, int)
                and isinstance(mapping, tuple)
                and mapping[0] == "replace"
            ):
                if prev_mapping > resolved_end and prev_mapping < num_truth_lines:
                    resolved_end = prev_mapping

        # Adjust for content between markers
        # If markers were part of larger replaced/deleted blocks, we need more precision
        # Check if any content between the markers was preserved or replaced
        content_preserved = False
        for idx in range(s_orig + 1, e_orig):
            if idx in conflict_to_truth_map:
                mapping = conflict_to_truth_map[idx]
                if isinstance(mapping, int) or (
                    isinstance(mapping, tuple) and mapping[0] == "replace"
                ):
                    content_preserved = True
                    break

        if not content_preserved:
            # If no content between markers was preserved, the resolved block might be empty
            # Check if consecutive deletions indicate an empty resolved block
            is_empty_resolution = False
            if (
                s_orig in conflict_to_truth_map
                and e_orig in conflict_to_truth_map
                and isinstance(conflict_to_truth_map[s_orig], tuple)
                and isinstance(conflict_to_truth_map[e_orig], tuple)
                and conflict_to_truth_map[s_orig][0] == "delete"
                and conflict_to_truth_map[e_orig][0] == "delete"
                and conflict_to_truth_map[s_orig][1] == conflict_to_truth_map[e_orig][1]
            ):

                # Both markers were deleted and map to the same position in truth
                is_empty_resolution = True
                resolved_start = conflict_to_truth_map[s_orig][1]
                resolved_end = resolved_start
                # print(
                #     f"  Detected empty resolution for block {cb.index}: both markers deleted at truth position {resolved_start}"
                # )

        # Sanity check and correction
        if resolved_end < resolved_start:
            # print(
            #     f"  Block {cb.index} (V3 Opcode): Detected completely deleted block (end={resolved_end} < start={resolved_start})."
            # )
            cb.labels.append("OPCODE_MAP_NEGATIVE_RANGE")

        cb.resolved_start_line = resolved_start  # Store 0-based index
        cb.resolved_end_line = resolved_end  # Store 0-based index

        # print(
        #     f"  Block {cb.index} (V3 Opcode): Orig Markers [{s_orig+1}, {e_orig+1}] -> Resolved Range [{resolved_start+1}-{resolved_end}] (0-based: [{resolved_start}, {resolved_end}])"
        # )

    # --- Step 5: Finalize Merged Content & Apply Judgers ---
    conflict_blocks_final: List[ConflictBlock] = []
    for cb_model in conflict_block_models:
        start = cb_model.resolved_start_line
        end = cb_model.resolved_end_line

        # Ensure final bounds are valid within truth_lines
        start = max(0, min(start, num_truth_lines))
        end = max(
            start, min(end, num_truth_lines)
        )  # Ensure end >= start and within bounds

        # Extract merged content using 0-based slice [start, end)
        cb_model.merged = truth_lines[start:end]

        # Prepare for judgers (1-based lines)
        judger_start_line = start + 1  # Convert 0-based index to 1-based line number
        judger_end_line = end  # End is exclusive in slice, inclusive for line numbers

        cb_for_judger = ConflictBlockModel(
            cb_model.index,
            cb_model.ours,
            cb_model.theirs,
            cb_model.base,
            cb_model.merged,
            [],  # Judger labels start fresh
            cb_model.start_line,  # 1-based original marker line
            cb_model.end_line,  # 1-based original marker line
            judger_start_line,  # 1-based resolved start
            judger_end_line,  # 1-based resolved end
        )

        current_judger_labels = []
        # 直接使用judgers，不需要检查它们是否在globals()中
        try:
            judgers = [
                ClassifierJudger(cb_for_judger),
                MergebotJudger(cb_for_judger),
            ]
            for judger in judgers:
                try:
                    current_judger_labels.extend(judger.judge())
                except Exception as e:
                    print(
                        f"    Judger {type(judger).__name__} failed for block {cb_model.index} (V3 Opcode): {e}"
                    )
                    current_judger_labels.append(f"{type(judger).__name__}_ERROR")
        except Exception as e:
            print(
                f"    Error initializing judgers for block {cb_model.index} (V3 Opcode): {e}"
            )
            current_judger_labels.append("JUDGER_INIT_ERROR")

        # Combine any opcode mapping labels with judger labels
        final_labels = cb_model.labels + current_judger_labels

        # print(
        #     f"    Block {cb_model.index} (V3 Opcode): Final resolved range [{judger_start_line}, {judger_end_line}] Lines: {len(cb_model.merged)}"
        # )

        conflict_blocks_final.append(
            ConflictBlock(
                cb_model.index,
                "\n".join(cb_model.ours),
                "\n".join(cb_model.theirs),
                "\n".join(cb_model.base),
                "\n".join(cb_model.merged),
                final_labels,
                cb_model.start_line,  # 1-based original start marker line
                cb_model.end_line,  # 1-based original end marker line
                judger_start_line,  # 1-based resolved start line number
                judger_end_line,  # 1-based resolved end line number
            )
        )

    # print(
    #     f"Finished V3 (Opcode) processing {len(conflict_blocks_final)} blocks for {merged_file}"
    # )
    return conflict_blocks_final
