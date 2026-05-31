python -m member_b2.evaluate --model results/dqn_r1_mlp_seed42/final_model.pt --reward sparse --encoder mlp --episodes 1000 --seed 999 --render-solutions 5
python -m member_b2.plot_curves --results-dir results/dqn_r1_mlp_seed42
python -m member_b2.evaluate --model results/dqn_r2_mlp_seed42/best_model.pt --reward shaped --encoder mlp --episodes 1000 --seed 999 --render-solutions 5
python -m member_b2.plot_curves --results-dir results/dqn_r2_mlp_seed42
