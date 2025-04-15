import glob
import json
import re
import requests
import os
import sys
import time
from os.path import join
from typing import List, Tuple, Dict
from git_service import process_conflict_blocks

base_url = "http://127.0.0.1:5000/resolve_conflict"

data = {"raw_a": "", "raw_b": "", "raw_base": ""}

result_dir = "./mergegen-output/c"
# 可能的后缀列表
extensions = ["cc", "h", "inl", "hpp", "C", "cpp", "cxx", "c", "cx"]

import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

CPP_LANGUAGE = Language(tscpp.language())

parser = Parser(CPP_LANGUAGE)


def remove_whitespace_and_compare(str1, str2):
    """Remove all whitespace characters (including spaces, tabs, and newlines) and compare."""
    str1_no_whitespace = re.sub(r"\s+", "", str1)
    str2_no_whitespace = re.sub(r"\s+", "", str2)
    return (
        str1_no_whitespace == str2_no_whitespace,
        str1_no_whitespace,
        str2_no_whitespace,
    )


def get_leading_context(
    conflicting_lines: List[str], start_line: int, context_size: int = 10
) -> str:
    """Get the leading context of the conflict block."""
    start_line = max(0, start_line - context_size + 1)
    return "\n".join(conflicting_lines[start_line : start_line + context_size])


def get_trailing_context(
    conflicting_lines: List[str], end_line: int, context_size: int = 10
) -> str:
    """Get the trailing context of the conflict block."""
    return "\n".join(conflicting_lines[end_line : end_line + context_size])


def ensure_dir_exists(dir_path: str):
    """Ensure the directory exists, create it if it doesn't."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def is_partial_syntax_valid(code: str, max_root_errors: int = 1) -> bool:
    tree = parser.parse(bytes(code, "utf-8"))
    root = tree.root_node
    
    errors = []
    
    # 递归收集所有 ERROR 节点
    def _collect_errors(node):
        if node.type == "ERROR":
            errors.append(node)
        for child in node.children:
            _collect_errors(child)
    
    _collect_errors(root)
    
    # 没有错误，直接通过
    if not errors:
        return True
    
    # 统计根节点直接子节点中的错误数量
    root_error_count = sum(1 for err in errors if err.parent == root)
    
    # 如果所有错误均位于根节点下，且数量不超过阈值
    if root_error_count == len(errors) and root_error_count <= max_root_errors:
        return True
    
    return False


def process_merge_scenario(
    ms_dir: str, ms_name: str, repo_name: str, output_dir: str
) -> Tuple[int, int, int, int, int, float]:
    """Process a single merge scenario and return statistics and execution time."""
    start_time = time.time()

    conflict_count = 0
    correct_count = 0
    wrong_count = 0
    hesitated_count = 0
    token_limited = 0
    rejected_count = 0

    # Create output directory for this merge scenario
    repo_output_dir = os.path.join(output_dir, repo_name)
    ms_output_dir = os.path.join(repo_output_dir, ms_name)
    ensure_dir_exists(ms_output_dir)

    patterns = [os.path.join(ms_dir, f"*.merged.{ext}") for ext in extensions]
    raw_conflict_files = []
    for pattern in patterns:
        raw_conflict_files.extend(glob.glob(pattern))

    for raw_conflict_file in raw_conflict_files:
        # Get the filename component of the path and replace "merged" with "truth"
        filename = os.path.basename(raw_conflict_file)
        merged_filename = filename.replace("merged", "truth")

        # Get the full path with the new filename
        merged_filepath = os.path.join(
            os.path.dirname(raw_conflict_file), merged_filename
        )

        if not os.path.exists(merged_filepath):
            print(
                f"!!!! warning: corresponding merged file not found for {raw_conflict_file} in {ms_dir}"
            )
            continue

        with open(merged_filepath, "r") as f:
            merged_content = f.read()
            merged_lines = merged_content.splitlines()

        with open(raw_conflict_file, "r") as f:
            conflict_content = f.read()
            conflict_lines = conflict_content.splitlines()

        conflict_blocks = process_conflict_blocks(conflict_content, merged_content, merged_filepath)
        conflict_count += len(conflict_blocks)

        conflicting_chunks = [
            {
                "a_contents": conflict_block.ours,
                "b_contents": conflict_block.theirs,
                "base_contents": conflict_block.base,
                "res_region": conflict_block.merged,
                "lookback": get_leading_context(
                    merged_lines, conflict_block.resolved_start_line, 1
                ),
                "lookahead": get_trailing_context(
                    merged_lines, conflict_block.resolved_end_line, 1
                ),
                "label": conflict_block.labels,
                "start_line": conflict_block.start_line - 1,
                "end_line": conflict_block.end_line - 1,
            }
            for conflict_block in conflict_blocks
        ]

        for conflict_chunk in conflicting_chunks:
            data.clear()
            data["raw_a"] = (
                conflict_chunk["lookback"]
                + conflict_chunk["a_contents"]
                + conflict_chunk["lookahead"]
            )
            data["raw_b"] = (
                conflict_chunk["lookback"]
                + conflict_chunk["b_contents"]
                + conflict_chunk["lookahead"]
            )
            data["raw_base"] = (
                conflict_chunk["lookback"]
                + conflict_chunk["base_contents"]
                + conflict_chunk["lookahead"]
            )

            response = requests.post(base_url, json=data)

            if response.status_code != 200:
                assert False, "illegal state"
            else:
                json_response = response.json()
                assert (
                    "code" in json_response and json_response["code"] == "200"
                ), "illegal response code"
                response_data = json_response["data"]
                resolved_content = response_data["resolved_content"]
                confidence = response_data["confidence"]
                hesitated = response_data["hesitated"]

                conflict_chunk["merge_gen_region"] = resolved_content
                conflict_chunk["confidence"] = confidence
                conflict_chunk["hesitated"] = hesitated

                if hesitated:
                    hesitated_count += 1
                else:
                    conflict_chunk["resolved"], res_no_space, truth_no_space = (
                        remove_whitespace_and_compare(
                            resolved_content if resolved_content is not None else "",
                            conflict_chunk["res_region"]
                        )
                    )

                    if conflict_chunk["resolved"]:
                        correct_count += 1
                    elif (
                        resolved_content is not None and
                        len(truth_no_space) > len(res_no_space) and
                        truth_no_space.startswith((res_no_space))
                    ):
                        token_limited += 1
                    elif resolved_content is not None and len(resolved_content) < 200 and not is_partial_syntax_valid(resolved_content):
                        print(f"<<<<<<<<<<<<<<<<<<<<< warning: partial syntax error in {raw_conflict_file}")
                        print(resolved_content)
                        print(">>>>>>>>>>>>>>>>>>>>>")
                        rejected_count += 1
                    else:
                        wrong_count += 1

        filename = os.path.basename(raw_conflict_file)
        mergegen_filename = filename.replace("merged", "mergegen") + ".json"

        # Save results to the output directory
        output_filepath = os.path.join(ms_output_dir, mergegen_filename)
        with open(output_filepath, "w") as f:
            json.dump(conflicting_chunks, f, indent=4)

    execution_time = time.time() - start_time

    return (
        conflict_count,
        correct_count,
        wrong_count,
        hesitated_count,
        token_limited,
        rejected_count,
        execution_time,
    )


def generate_mergegen_output(
    repo: str, test_dir: str, output_dir: str, timing_file
) -> Tuple[int, int, int, int, int]:
    """Generate mergegen output for a repository and record timing information."""
    total_conflict_block_count = 0
    total_correct_count = 0
    total_wrong_count = 0
    total_hesitated_count = 0
    total_token_limited = 0
    total_rejected_count = 0

    # Ensure repo output directory exists
    repo_output_dir = os.path.join(output_dir, repo)
    ensure_dir_exists(repo_output_dir)

    repo_dir = os.path.join(test_dir, repo)
    mss = os.listdir(repo_dir)

    for ms in mss:
        ms_dir = os.path.join(repo_dir, ms)
        if not os.path.isdir(ms_dir):
            continue

        # Process this merge scenario
        (
            conflict_count,
            correct_count,
            wrong_count,
            hesitated_count,
            token_limited,
            rejected_count,
            execution_time,
        ) = process_merge_scenario(ms_dir, ms, repo, output_dir)

        # Update totals
        total_conflict_block_count += conflict_count
        total_correct_count += correct_count
        total_wrong_count += wrong_count
        total_hesitated_count += hesitated_count
        total_token_limited += token_limited
        total_rejected_count += rejected_count

        # Record timing information
        timing_file.write(f"{repo},{ms},{execution_time:.4f}\n")
        timing_file.flush()  # Ensure data is written immediately

        # Print progress
        print(
            f"  Processed {ms}: {conflict_count} conflicts in {execution_time:.4f} seconds"
        )

    return (
        total_conflict_block_count,
        total_correct_count,
        total_wrong_count,
        total_hesitated_count,
        total_token_limited,
        total_rejected_count,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mergegen.py <test_dir> [output_dir] [timing_file] [summary_file]")
        exit(1)

    test_dir = sys.argv[1]
    assert os.path.exists(test_dir), "test dir must exist"

    # Use provided output directory or default
    output_dir = sys.argv[2] if len(sys.argv) > 2 else result_dir
    ensure_dir_exists(output_dir)

    # Use provided timing file or default
    timing_file_path = sys.argv[3] if len(sys.argv) > 3 else "mergegen_timing.csv"
    
    # Use provided summary file or default
    summary_file_path = sys.argv[4] if len(sys.argv) > 4 else "mergegen_summary.csv"

    # Create timing file with header
    with open(timing_file_path, "w") as timing_file:
        timing_file.write("repo,merge_scenario,execution_time_seconds\n")
        
        # Create summary file with header
        with open(summary_file_path, "w") as summary_file:
            summary_file.write("repo,total_conflicts,correct,wrong,hesitated,token_limited,rejected,precision,accuracy\n")
            
            # Process all repositories
            for repo in os.listdir(test_dir):
                repo_path = os.path.join(test_dir, repo)
                if not os.path.isdir(repo_path):
                    continue

                print(f"Processing repo: {repo}")
                start_time = time.time()

                # Generate mergegen output for this repository
                result = generate_mergegen_output(repo, test_dir, output_dir, timing_file)
                total_conflicts, correct, wrong, hesitated, token_limited, rejected = result
                
                # Calculate accuracy
                accuracy = (correct + token_limited) / total_conflicts if total_conflicts > 0 else 0
                # Calculate precision
                precision_denominator = total_conflicts - rejected - hesitated
                precision = (correct + token_limited) / precision_denominator if precision_denominator > 0 else 0

                # Write summary for this repository
                summary_file.write(f"{repo},{total_conflicts},{correct},{wrong},{hesitated},{token_limited},{rejected},{precision:.4f},{accuracy:.4f}\n")
                summary_file.flush()  # Ensure data is written immediately

                # Calculate and print repository execution time
                repo_execution_time = time.time() - start_time
                print(f"Execution time for {repo}: {repo_execution_time:.4f} seconds")
                print("Total conflicts, correct, wrong, hesitated, token-limited, rejected:", result)
                print("-" * 80)
