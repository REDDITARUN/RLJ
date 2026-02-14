"""End-to-end smoke test + post-training inference demo.

Runs a tiny experiment (2 optimizer steps) for each encoder to verify:
  1. Model builds correctly (encoder + projector + LoRA LLM)
  2. Forward pass produces a loss
  3. Backward pass + optimizer step works
  4. Outputs are saved (projector.pt, lora_adapter/, loss_history.json)
  5. Saved weights can be reloaded
  6. Inference (generation) works with the trained bridge

Usage:
    python test.py                  # test all 3 encoders
    python test.py --encoder clip   # test one encoder
"""

import gc
import json
import os
import sys
import time

import torch

from configs import ExperimentConfig, get_encoder_configs
from data import CauldronDataset, VLMCollator
from models import StitchedVLM
from models.projector import Projector
from models.vision_encoders import ENCODER_META
from train import run_experiment, _get_device, _resolve_dtype
from utils import LossTracker, plot_convergence


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

PASS = "PASS"
FAIL = "FAIL"


def _status(ok: bool) -> str:
    return PASS if ok else FAIL


def _section(title: str):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'=' * 60}", flush=True)


# --------------------------------------------------------------------------- #
#  Test 1: Model construction
# --------------------------------------------------------------------------- #


def test_model_build(config: ExperimentConfig, device: torch.device, dtype: torch.dtype):
    """Verify the stitched model builds and reports correct param counts."""
    _section(f"TEST 1: Model Build  [{config.encoder_name.upper()}]")

    model = StitchedVLM(
        encoder_model_id=config.encoder_model_id,
        encoder_type=config.encoder_type,
        llm_model_id=config.llm_model_id,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        dtype=dtype,
    )
    model = model.to(device)

    trainable = model.count_trainable_parameters()
    total = model.count_total_parameters()

    print(f"  Trainable:  {trainable:,}", flush=True)
    print(f"  Total:      {total:,}", flush=True)

    ok = trainable > 0 and trainable < total
    print(f"  [{_status(ok)}] Trainable params exist and < total", flush=True)

    # Check encoder is frozen
    encoder_grads = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    ok_frozen = encoder_grads == 0
    print(f"  [{_status(ok_frozen)}] Encoder is fully frozen ({encoder_grads} grad params)", flush=True)

    return model, ok and ok_frozen


# --------------------------------------------------------------------------- #
#  Test 2: Forward + Backward
# --------------------------------------------------------------------------- #


def test_forward_backward(model: StitchedVLM, config: ExperimentConfig, device: torch.device):
    """Verify forward pass produces loss and backward pass computes gradients."""
    _section(f"TEST 2: Forward + Backward  [{config.encoder_name.upper()}]")

    # Load a tiny batch
    dataset = CauldronDataset(
        dataset_name=config.dataset_name,
        subset=config.dataset_subset,
        max_samples=8,
    )
    collator = VLMCollator(
        image_processor=model.encoder.processor,
        tokenizer=model.tokenizer,
        max_seq_len=config.max_seq_len,
    )

    batch_items = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = collator(batch_items)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward
    model.train()
    outputs = model(**batch)
    loss = outputs["loss"]

    ok_loss = loss is not None and not torch.isnan(loss) and loss.item() > 0
    print(f"  Loss value: {loss.item():.4f}", flush=True)
    print(f"  [{_status(ok_loss)}] Loss is valid (not NaN, > 0)", flush=True)

    # Backward
    loss.backward()

    proj_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.projector.parameters()
    )
    print(f"  [{_status(proj_grad)}] Projector has gradients", flush=True)

    model.zero_grad()
    return ok_loss and proj_grad


# --------------------------------------------------------------------------- #
#  Test 3: Mini training run (2 steps)
# --------------------------------------------------------------------------- #


def test_training(config: ExperimentConfig, device: torch.device):
    """Run 2 optimizer steps and verify loss is tracked."""
    _section(f"TEST 3: Mini Training (2 steps)  [{config.encoder_name.upper()}]")

    # Override config for a tiny run
    config.num_steps = 2
    config.max_samples = 32
    config.batch_size = 2
    config.gradient_accumulation_steps = 2
    config.warmup_steps = 0
    config.save_dir = "test_outputs"

    tracker = run_experiment(config, device=device, use_wandb=False, push_to_hub=False)
    summary = tracker.summary()

    ok_steps = summary["total_steps"] == 2
    ok_loss = summary["final_loss"] > 0
    print(f"  Steps completed: {summary['total_steps']}", flush=True)
    print(f"  Final loss:      {summary['final_loss']:.4f}", flush=True)
    print(f"  [{_status(ok_steps)}] Completed 2 steps", flush=True)
    print(f"  [{_status(ok_loss)}] Loss is positive", flush=True)

    return tracker, ok_steps and ok_loss


# --------------------------------------------------------------------------- #
#  Test 4: Saved outputs
# --------------------------------------------------------------------------- #


def test_saved_outputs(config: ExperimentConfig):
    """Verify all expected files were saved."""
    _section(f"TEST 4: Saved Outputs  [{config.encoder_name.upper()}]")

    save_dir = os.path.join(config.save_dir, config.encoder_name)

    files_to_check = [
        "loss_history.json",
        "projector.pt",
        "config.json",
    ]
    dirs_to_check = [
        "lora_adapter",
    ] if config.use_lora else []

    all_ok = True
    for f in files_to_check:
        path = os.path.join(save_dir, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  [{_status(exists)}] {f}  ({size:,} bytes)", flush=True)
        all_ok = all_ok and exists

    for d in dirs_to_check:
        path = os.path.join(save_dir, d)
        exists = os.path.isdir(path)
        n_files = len(os.listdir(path)) if exists else 0
        print(f"  [{_status(exists)}] {d}/  ({n_files} files)", flush=True)
        all_ok = all_ok and exists

    return all_ok


# --------------------------------------------------------------------------- #
#  Test 5: Reload weights
# --------------------------------------------------------------------------- #


def test_reload(config: ExperimentConfig, device: torch.device, dtype: torch.dtype):
    """Verify saved projector and LoRA can be reloaded."""
    _section(f"TEST 5: Reload Weights  [{config.encoder_name.upper()}]")

    save_dir = os.path.join(config.save_dir, config.encoder_name)

    # Reload projector
    vision_dim, _ = ENCODER_META[config.encoder_type]
    proj = Projector(vision_dim=vision_dim, llm_dim=896).to(dtype).to(device)
    proj_path = os.path.join(save_dir, "projector.pt")
    proj.load_state_dict(torch.load(proj_path, map_location=device, weights_only=True))
    ok_proj = True
    print(f"  [{_status(ok_proj)}] Projector reloaded from {proj_path}", flush=True)

    # Reload LoRA
    ok_lora = True
    if config.use_lora:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id, dtype=dtype
        ).to(device)
        lora_path = os.path.join(save_dir, "lora_adapter")
        lora_llm = PeftModel.from_pretrained(base_llm, lora_path)
        ok_lora = lora_llm is not None
        print(f"  [{_status(ok_lora)}] LoRA adapter reloaded from {lora_path}/", flush=True)
        del base_llm, lora_llm

    del proj
    return ok_proj and ok_lora


# --------------------------------------------------------------------------- #
#  Test 6: Inference / Generation
# --------------------------------------------------------------------------- #


def test_inference(config: ExperimentConfig, device: torch.device, dtype: torch.dtype):
    """Load trained model and generate text from an image."""
    _section(f"TEST 6: Inference  [{config.encoder_name.upper()}]")

    save_dir = os.path.join(config.save_dir, config.encoder_name)

    # Build model fresh
    model = StitchedVLM(
        encoder_model_id=config.encoder_model_id,
        encoder_type=config.encoder_type,
        llm_model_id=config.llm_model_id,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        dtype=dtype,
    )

    # Load trained projector
    proj_path = os.path.join(save_dir, "projector.pt")
    model.projector.load_state_dict(
        torch.load(proj_path, map_location="cpu", weights_only=True)
    )

    # Load trained LoRA
    if config.use_lora:
        from peft import PeftModel
        lora_path = os.path.join(save_dir, "lora_adapter")
        # Merge LoRA for cleaner inference (optional)
        model.llm = PeftModel.from_pretrained(model.llm.base_model.model, lora_path)

    model = model.to(device)
    model.eval()

    # Get a sample image
    dataset = CauldronDataset(
        dataset_name=config.dataset_name,
        subset=config.dataset_subset,
        max_samples=4,
    )
    sample = dataset[0]
    image = sample["image"]
    question = sample["instruction"]

    print(f"  Question: {question}", flush=True)

    # Preprocess image
    pixel_values = model.encoder.preprocess([image])["pixel_values"].to(device)

    # Encode image -> project
    with torch.no_grad():
        vision_tokens = model.encoder(pixel_values)
        visual_embeds = model.projector(vision_tokens.to(dtype))

    # Tokenize the instruction
    tok = model.tokenizer(question, return_tensors="pt").to(device)

    # Get text embeddings
    from peft import PeftModel as _PM
    if isinstance(model.llm, _PM):
        embed_layer = model.llm.model.model.embed_tokens
    else:
        embed_layer = model.llm.model.embed_tokens

    text_embeds = embed_layer(tok["input_ids"])

    # Concat [visual | text]
    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        generated_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=50,
            do_sample=False,
        )

    generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"  Generated: {generated_text}", flush=True)

    ok = len(generated_text.strip()) > 0
    print(f"  [{_status(ok)}] Model generated non-empty text", flush=True)

    del model
    return ok


# --------------------------------------------------------------------------- #
#  Test 7: Convergence plot
# --------------------------------------------------------------------------- #


def test_plot(trackers: dict, save_dir: str):
    """Verify the convergence plot can be generated."""
    _section("TEST 7: Convergence Plot")

    save_path = os.path.join(save_dir, "test_convergence.png")
    fig = plot_convergence(trackers, save_path=save_path)

    ok = os.path.isfile(save_path)
    print(f"  [{_status(ok)}] Plot saved to {save_path}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def run_all_tests(encoder_filter: str = None):
    """Run all tests, return total pass/fail count."""
    device = _get_device()
    dtype = _resolve_dtype("bfloat16", device)

    configs = get_encoder_configs()
    if encoder_filter:
        configs = [c for c in configs if c.encoder_name == encoder_filter]

    results = []
    trackers = {}

    for cfg in configs:
        _section(f"TESTING ENCODER: {cfg.encoder_name.upper()}")
        print(f"  Model: {cfg.encoder_model_id}", flush=True)
        print(f"  Device: {device}  Dtype: {dtype}", flush=True)

        # Test 1: Build
        model, ok1 = test_model_build(cfg, device, dtype)
        results.append(("Build", cfg.encoder_name, ok1))

        # Test 2: Forward + Backward
        ok2 = test_forward_backward(model, cfg, device)
        results.append(("Forward/Backward", cfg.encoder_name, ok2))

        # Free model before training builds a new one
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test 3: Mini training
        tracker, ok3 = test_training(cfg, device)
        trackers[cfg.encoder_name] = tracker
        results.append(("Training", cfg.encoder_name, ok3))

        # Test 4: Saved outputs
        ok4 = test_saved_outputs(cfg)
        results.append(("Saved Outputs", cfg.encoder_name, ok4))

        # Test 5: Reload
        ok5 = test_reload(cfg, device, dtype)
        results.append(("Reload", cfg.encoder_name, ok5))

        # Test 6: Inference
        ok6 = test_inference(cfg, device, dtype)
        results.append(("Inference", cfg.encoder_name, ok6))

        # Cleanup between encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Test 7: Plot (if multiple encoders)
    if len(trackers) > 1:
        ok7 = test_plot(trackers, "test_outputs")
        results.append(("Plot", "all", ok7))

    # ---- Final Report ----
    _section("FINAL REPORT")
    passed = sum(1 for _, _, ok in results if ok)
    failed = sum(1 for _, _, ok in results if not ok)

    for test_name, enc, ok in results:
        mark = PASS if ok else FAIL
        print(f"  [{mark}]  {enc:8s}  {test_name}", flush=True)

    print(f"\n  Total: {passed} passed, {failed} failed out of {len(results)}", flush=True)
    print("=" * 60, flush=True)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Benchmark End-to-End Test")
    parser.add_argument(
        "--encoder", type=str, default=None,
        choices=["vit", "clip", "ijepa"],
        help="Test a single encoder (default: test all three).",
    )
    args = parser.parse_args()

    success = run_all_tests(encoder_filter=args.encoder)
    sys.exit(0 if success else 1)
