# Demonstration Results

## Generated Images

### Training and Performance
1. **training_convergence.png** - Training convergence curves showing reward and loss metrics
2. **performance_summary.png** - Performance summary card with 100% success rate

### Trajectory Visualizations
3. **trajectory_comparison.png** - Multi-trajectory comparison (10 episodes)
4. **trajectory_detailed.png** - Detailed trajectory analysis with velocity and cross-track error

## Important Notes

### Success Rate Discrepancy

The evaluation results (`final_evaluation.json`) show **100% success rate** (512/512 episodes), but the trajectory visualizations show lower success rates (20% for comparison, 0% for detailed). This is due to **different tolerance settings**:

**Evaluation Tolerances** (used in final_evaluation.json):
- Position: 0.8m
- Velocity: 0.1m/s
- Heading: 10° (0.175 rad)
- Yaw rate: 0.05 rad/s

**Visualization Tolerances** (current Level 2 defaults):
- Position: 0.3m
- Velocity: 0.05m/s
- Heading: 7° (0.122 rad)
- Yaw rate: 0.03 rad/s

The visualization uses **stricter tolerances** than the evaluation, which explains the lower success rates. The model performs well under the evaluation criteria but shows room for improvement under stricter requirements.

## Configuration

All visualizations use:
- Curriculum Level: 2
- Current: Enabled (0.075 m/s, fixed mode)
- Wind: Enabled (1.2 m/s, fixed mode)
- Model: `per_sac_otter_checkpoint_400.pth`
- Observation Space: 30D (includes disturbance information)

## Files

- `final_evaluation.json` - Evaluation results with 100% success rate
- `training_convergence.png` - Training metrics
- `performance_summary.png` - Performance summary
- `trajectory_comparison.png` - 10-episode trajectory comparison
- `trajectory_detailed.png` - Detailed 5-episode analysis
