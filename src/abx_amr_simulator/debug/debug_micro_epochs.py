"""
Debug training with micro-epochs while printing transitions.

Usage examples:

    python experiments/debug_micro_epochs.py \
        --config experiments/configs/agent_algorithm/ppo_baseline.yaml \
        --micro-steps 10 --chunks 50 --print-obs 0

This runs SB3 training in small bursts (micro-epochs), wrapping the env with
StepPrinter to give per-step visibility without committing to a long run.
Keep this for short debug sessions; printing will slow training significantly.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from functools import partial

import pdb # for debugging purposes

import numpy as np
from stable_baselines3 import PPO, A2C

from abx_amr_simulator.utils import (
    load_config,
    create_reward_calculator,
    create_patient_generator,
    create_environment,
    create_agent,
    create_run_directory,
    save_training_config,
    # plot utility
    plot_metrics_trained_agent,
)
from abx_amr_simulator.wrappers import StepPrinter
from abx_amr_simulator.formatters import abx_amr_step_formatter


def main():
    parser = argparse.ArgumentParser(description="Micro-epoch training with transition printing")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--micro-steps", type=int, default=10, help="Timesteps per learn() call")
    parser.add_argument("--chunks", type=int, default=50, help="Number of micro-epochs to run")
    parser.add_argument("--print-obs", type=int, default=0, help="Print observations (0/1)")
    parser.add_argument(
        "--max-print-steps",
        type=int,
        default=200,
        help="Max printed steps per debug episode (StepPrinter)",
    )
    parser.add_argument(
        "--info-keys",
        type=str,
        default="",
        help="Comma-separated subset of info keys to print (blank prints key list only)",
    )
    parser.add_argument(
        "--diag-every",
        type=int,
        default=5,
        help="Generate diagnostic figures every N micro-epochs",
    )
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Save StepPrinter output to a log file under the run directory",
    )
    parser.add_argument(
        "--log-all-steps",
        action="store_true",
        help="When logging, record all steps even if max-print-steps suppresses stdout",
    )
    parser.add_argument(
        "--use-formatter",
        action="store_true",
        help="Use human-readable step formatter (abx_amr_step_formatter) instead of default format",
    )
    parser.add_argument(
        "--per-chunk-figs",
        action="store_true",
        help="When set, save diagnostics into per-chunk subfolders (figures/chunk_XXXX)",
    )
    parser.add_argument(
        "--tb",
        action="store_true",
        help="Enable TensorBoard logging to scripts/tb_logs",
    )

    # Alternate cycles mode: evaluation-only plotting alternating with training bursts
    parser.add_argument(
        "--mode",
        type=str,
        choices=["micro", "alternate"],
        default="micro",
        help="Run standard micro-epochs or alternate debug/train cycles",
    )
    parser.add_argument(
        "--alt-cycles",
        type=int,
        default=5,
        help="Number of debug/train cycles in alternate mode",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1000,
        help="Training timesteps per cycle in alternate mode, that is how many steps the agent trains on between evaluations via micro-epochs",
    )
    parser.add_argument(
        "--eval-print",
        action="store_true",
        help="Print transitions during evaluation-only debug episodes (alternate mode)",
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = os.path.join(str(PROJECT_ROOT), args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)

    # Create reward calculator, patient generator and env
    reward_calculator = create_reward_calculator(config)
    patient_generator = create_patient_generator(config)
    env = create_environment(config=config, reward_calculator=reward_calculator, patient_generator=patient_generator)

    # Seed env and numpy for determinism; if the user didn't pass in a seed as an argument, get it from config:
    if args.seed is None:
        args.seed = config.get("training", {}).get("seed", 42)
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    
    print(f"Seed: {args.seed}")

    # Create run directory and save config
    run_dir, _ = create_run_directory(config, str(PROJECT_ROOT))
    save_training_config(config, run_dir)

    # Shared TB log path
    tb_path = os.path.join(str(PROJECT_ROOT), "scripts", "tb_logs") if args.tb else None

    if args.mode == "micro":
        # Wrap env with StepPrinter for training visibility
        info_keys = [k.strip() for k in args.info_keys.split(",") if k.strip()] if args.info_keys else None
        log_path = os.path.join(run_dir, "step_printer.log") if args.log_steps else None
        
        # Get antibiotic mapping from env and bind to formatter
        # abx_name_to_action_mapping = env.get_antibiotic_to_action_mapping()
        action_index_to_abx_name_mapping = env.get_action_to_antibiotic_mapping()
        step_fmt = partial(abx_amr_step_formatter, action_idx_to_abx_name=action_index_to_abx_name_mapping) if args.use_formatter else None
        env = StepPrinter(
            env,
            print_obs=bool(args.print_obs),
            formatter=step_fmt,
            print_info_keys=info_keys,
            max_print_steps=args.max_print_steps if args.max_print_steps else args.micro_steps * 2,
            prefix="[DBG] ",
            float_precision=3,
            log_path=log_path,
            log_all_steps=args.log_all_steps,
        )

        # Create agent on the wrapped env
        agent = create_agent(config, env, tb_log_path=tb_path)

        # Micro-epochs: repeatedly call learn() in small increments
        print(
            f"Starting micro-epochs: micro_steps={args.micro_steps}, chunks={args.chunks}, seed={args.seed}"
        )
        total = 0
        for i in range(args.chunks):
            agent.learn(
                total_timesteps=args.micro_steps,
                reset_num_timesteps=False,
                tb_log_name=config.get("run_name", "micro_debug"),
                progress_bar=False,
            )
            total += args.micro_steps
            print(f"[DBG] completed chunk {i+1}/{args.chunks} (total_timesteps={total})")

            # Produce diagnostic figures at requested cadence
            if (i + 1) % max(1, args.diag_every) == 0:
                try:
                    if args.per_chunk_figs:
                        # Save into per-chunk subfolder under run_dir
                        chunk_dir = os.path.join(run_dir, f"chunk_{i+1:04d}")
                        os.makedirs(chunk_dir, exist_ok=True)
                        env.log_header(f"\n{'='*60}\nDiagnostics Chunk {i+1}\n{'='*60}")
                        print(
                            f"[DBG] generating diagnostics at chunk {i+1} -> {chunk_dir}/figures"
                        )
                        plot_metrics_trained_agent(agent, env, chunk_dir, deterministic=True)
                    else:
                        env.log_header(f"\n{'='*60}\nDiagnostics Chunk {i+1}\n{'='*60}")
                        print(f"[DBG] generating diagnostics at chunk {i+1} -> {run_dir}/figures")
                        plot_metrics_trained_agent(agent, env, run_dir, deterministic=True)
                except Exception as e:
                    print(f"[DBG] diagnostics error: {type(e).__name__}: {e}")

        print("[DBG] Micro-epoch debug run complete.")
        
        # Save final model for micro mode
        final_model_path = os.path.join(run_dir, 'checkpoints', 'final_model')
        agent.save(final_model_path)
        print(f"\nFinal model (end of training) saved to: {final_model_path}.zip")

    elif args.mode == "alternate":
        # In alternate mode, keep training env unwrapped; use a StepPrinter-wrapped env only for evaluation/plotting
        agent = create_agent(config, env, tb_log_path=tb_path)

        eval_info_keys = [k.strip() for k in args.info_keys.split(",") if k.strip()] if args.info_keys else None
        eval_log_path = os.path.join(run_dir, "eval_step_printer.log") if args.log_steps else None
        
        # Get antibiotic mapping from env and bind to formatter (index -> name)
        action_index_to_abx_name_mapping = env.get_action_to_antibiotic_mapping()
        eval_step_fmt = partial(abx_amr_step_formatter, action_idx_to_abx_name=action_index_to_abx_name_mapping) if args.use_formatter else None
        
        eval_env = StepPrinter(
            env,
            print_obs=bool(args.print_obs) if args.eval_print else False,
            formatter=eval_step_fmt,
            print_info_keys=eval_info_keys,
            max_print_steps=args.max_print_steps,
            prefix="[DBG] ",
            float_precision=3,
            log_path=eval_log_path,
            log_all_steps=args.log_all_steps,
            mirror_stdout=bool(args.eval_print),
        )

        print(
            f"Starting alternate cycles: cycles={args.alt_cycles}, train_steps/cycle={args.train_steps}, seed={args.seed}"
        )
        total = 0
        for cyc in range(1, args.alt_cycles + 1):
            # Evaluation-only plotting for current policy
            try:
                cycle_dir = os.path.join(run_dir, f"cycle_{cyc:03d}")
                os.makedirs(cycle_dir, exist_ok=True)
                eval_env.log_header(f"\n{'='*60}\nEvaluation Cycle {cyc}\n{'='*60}")
                print(f"[DBG] evaluation diagnostics for cycle {cyc} -> {cycle_dir}/figures")
                plot_metrics_trained_agent(agent, eval_env, cycle_dir, deterministic=True)
            except Exception as e:
                print(f"[DBG] evaluation diagnostics error: {type(e).__name__}: {e}")

            # Training burst
            agent.learn(
                total_timesteps=args.train_steps,
                reset_num_timesteps=False,
                tb_log_name=config.get("run_name", "alt_debug"),
                progress_bar=False,
            )
            total += args.train_steps
            print(f"[DBG] completed training for cycle {cyc}/{args.alt_cycles} (total_timesteps={total})")

        # Final evaluation after last cycle
        try:
            final_dir = os.path.join(run_dir, f"cycle_{args.alt_cycles:03d}_final")
            os.makedirs(final_dir, exist_ok=True)
            eval_env.log_header(f"\n{'='*60}\nFinal Evaluation\n{'='*60}")
            print(f"[DBG] final evaluation diagnostics -> {final_dir}/figures")
            plot_metrics_trained_agent(agent, eval_env, final_dir, deterministic=True)
        except Exception as e:
            print(f"[DBG] final diagnostics error: {type(e).__name__}: {e}")

        print("[DBG] Alternate cycles run complete.")

    # Save final model regardless of mode
    # final_model: model at end of training (useful for debugging)
    final_model_path = os.path.join(run_dir, 'checkpoints', 'final_model')
    agent.save(final_model_path)
    print(f"\nFinal model (end of training) saved to: {final_model_path}.zip")
    
    # Generate final diagnostics comparing best vs final agent
    algorithm = config.get('algorithm', 'PPO')
    algorithm_map = {'PPO': PPO, 'A2C': A2C}
    AgentClass = algorithm_map.get(algorithm, PPO)
    
    # Determine which env to use for diagnostics
    diagnostic_env = eval_env if args.mode == "alternate" else env
    
    # Plot metrics for best agent (highest eval reward during training)
    best_model_checkpoint = os.path.join(run_dir, 'checkpoints', 'best_model.zip')
    if os.path.exists(best_model_checkpoint):
        print("\n" + "="*70)
        print("Generating diagnostics for BEST agent (highest eval reward)")
        print("="*70)
        best_agent = AgentClass.load(best_model_checkpoint, env=diagnostic_env)
        best_figs_dir = os.path.join(run_dir, 'figures_best_agent')
        os.makedirs(best_figs_dir, exist_ok=True)
        plot_metrics_trained_agent(model=best_agent, env=diagnostic_env, experiment_folder=best_figs_dir, deterministic=True)
    else:
        print("\n[DBG] Warning: best_model.zip not found, skipping best agent diagnostics")

if __name__ == "__main__":
    main()
