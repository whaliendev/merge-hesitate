clean:
	rm -rf RAW_DATA/split
	rm -rf GIT_MERGE_FILES
	rm -rf PROCESSED
	rm -rf PREPROCESSED

split:
	uv run scripts/split_dataset.py --data_path RAW_DATA/cpp.json --output_dir RAW_DATA/split

preprocess:
	uv run scripts/preprocess_dataset.py

data:
	uv run scripts/split_dataset.py --data_path RAW_DATA/cpp.json --output_dir RAW_DATA/split
	uv run scripts/preprocess_dataset.py
