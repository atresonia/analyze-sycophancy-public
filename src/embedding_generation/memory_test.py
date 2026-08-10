import torch, time
from mlx_embeddings.utils import load
import mlx.core as mx
import os
os.environ["HF_TRUST_REMOTE_CODE"] = "true"

def main():
    model_name = "ncorder/llama-embed-nemotron-8b-mlx-8bit"
    model, tokenizer = load(model_name)

    seq_len = 512
    text = "test sentence " * (seq_len // 3)

    for batch_size in [4, 8, 16, 32, 64]:
        try:
            inputs = tokenizer([text] * batch_size, return_tensors="np", padding="max_length", 
                        truncation=True, max_length=seq_len)
            input_ids = mx.array(inputs["input_ids"])
            attention_mask = mx.array(inputs["attention_mask"])

            mx.clear_cache()
            mx.eval(input_ids, attention_mask)

            t0 = time.time()
            for _ in range(3):
                out = model(input_ids, attention_mask)
                mx.eval(out.text_embeds)
            
            elapsed = (time.time() - t0) / 3

            peak_mem = mx.get_peak_memory() / 1e9
            throughput = batch_size / elapsed
            print(f"Batch size: {batch_size}, Throughput: {throughput:.2f} tokens/s, "
                f"Peak memory: {peak_mem:.2f} GB")
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            break

if __name__ == "__main__":
    main()