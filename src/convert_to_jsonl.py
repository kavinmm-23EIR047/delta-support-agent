"""
Converts large formatted JSON array delta_reconstructed_threads.json
to line-by-line JSONL delta_reconstructed_threads_sample.jsonl
using chunked regex object splitting to avoid memory allocation spikes.
"""
import os
import json

ARTIFACTS_DIR = r"d:\hiver_task\artifacts"
SRC_JSON = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads.json")
OUT_JSONL = os.path.join(ARTIFACTS_DIR, "delta_reconstructed_threads_sample.jsonl")

def convert():
    print(f"Streaming {SRC_JSON} into {OUT_JSONL}...")
    count = 0
    buffer = ""
    
    with open(SRC_JSON, "r", encoding="utf-8") as f_in, open(OUT_JSONL, "w", encoding="utf-8") as f_out:
        # Read in 64KB chunks
        for chunk in iter(lambda: f_in.read(65536), ""):
            buffer += chunk
            # Split by top-level object closing and opening
            while True:
                # Find start of object
                start = buffer.find('{\n    "thread_id":')
                if start == -1:
                    start = buffer.find('{\n  "thread_id":')
                if start == -1:
                    break
                # Find matching closing bracket before next object
                next_start = buffer.find('{\n    "thread_id":', start + 10)
                if next_start == -1:
                    next_start = buffer.find('{\n  "thread_id":', start + 10)
                if next_start == -1:
                    # Look for end of array
                    end_arr = buffer.find('\n]', start)
                    if end_arr != -1:
                        obj_str = buffer[start:end_arr].strip().rstrip(',')
                        try:
                            obj = json.loads(obj_str)
                            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            count += 1
                        except:
                            pass
                        buffer = ""
                        break
                    else:
                        # Need more chunk
                        break
                else:
                    obj_str = buffer[start:next_start].strip().rstrip(',')
                    try:
                        obj = json.loads(obj_str)
                        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        count += 1
                        if count >= 6000:
                            print(f"Sample limit reached: {count:,} items.")
                            return
                    except Exception as e:
                        pass
                    buffer = buffer[next_start:]
                    
    print(f"Finished converting {count:,} objects to {OUT_JSONL}")

if __name__ == "__main__":
    convert()
