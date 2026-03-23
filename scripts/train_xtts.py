"""Fine-tuning de XTTS-v2 basado en la receta oficial de Coqui."""

from __future__ import annotations

import os
from pathlib import Path

from tts_project.config import ROOT_DIR
from tts_project.xtts import load_xtts_config


def main() -> None:
    """Ejecuta fine-tuning de XTTS usando el dataset exportado."""
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
        from TTS.utils.manage import ModelManager
    except ImportError as error:  # noqa: BLE001
        raise ImportError(
            "Faltan dependencias de XTTS. Usa requirements-xtts-colab.txt en Colab o un entorno GPU."
        ) from error

    config = load_xtts_config()
    dataset_path = ROOT_DIR / config["dataset_path"]
    output_path = ROOT_DIR / config["output_path"]
    checkpoints_path = output_path / "base_model"
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    dvae_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    mel_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    tokenizer_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    xtts_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    files = [mel_link, dvae_link, tokenizer_link, xtts_link]
    ModelManager._download_model_files(files, str(checkpoints_path), progress_bar=True)

    base_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=config["dataset_name"],
        path=str(dataset_path),
        meta_file_train=str(dataset_path / config["metadata_file"]),
        language=config["language"],
    )
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        max_wav_length=int(22050 * config["max_audio_seconds"]),
        max_text_length=config["max_text_length"],
        mel_norm_file=str(checkpoints_path / "mel_stats.pth"),
        dvae_checkpoint=str(checkpoints_path / "dvae.pth"),
        xtts_checkpoint=str(checkpoints_path / "model.pth"),
        tokenizer_file=str(checkpoints_path / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    trainer_config = GPTTrainerConfig(
        output_path=str(output_path),
        model_args=model_args,
        run_name=config["run_name"],
        project_name=config["project_name"],
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=config["batch_size"],
        batch_group_size=24,
        eval_batch_size=config["batch_size"],
        num_loader_workers=config["num_loader_workers"],
        eval_split_max_size=config["eval_split_max_size"],
        print_step=25,
        plot_step=100,
        log_model_step=500,
        save_step=2000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=config["learning_rate"],
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": sentence,
                "speaker_wav": [str(ROOT_DIR / config["speaker_reference"])],
                "language": config["language"],
            }
            for sentence in config["test_sentences"]
        ],
    )

    model = GPTTrainer.init_from_config(trainer_config)
    train_samples, eval_samples = load_tts_samples(
        [base_dataset],
        eval_split=True,
        eval_split_max_size=trainer_config.eval_split_max_size,
        eval_split_size=trainer_config.eval_split_size,
    )
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=True, grad_accum_steps=config["grad_accum_steps"]),
        trainer_config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    main()
