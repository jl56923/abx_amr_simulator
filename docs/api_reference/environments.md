# Environment API Reference (PLACEHOLDER)

**Status**: Placeholder — Content to be developed in discussion with user

## Intended Scope

This document will provide comprehensive reference material for the primary RL environment (`ABXAMREnv`) and guidelines for implementing custom environments.

### Proposed Sections

1. **ABXAMREnv Class Reference**
   - Complete API documentation including all methods and parameters
   - Design rationale for example observation/action space structure
   - Integration diagram showing how environment orchestrates rewards, patient generation, and AMR dynamics
   - Common configuration patterns from experiments

2. **Observation & Action Space Construction**
   - Details on how observations are assembled (patient attributes + AMR levels + optional step count)
   - Observation space sizing and dtype requirements
   - Action space structure (MultiDiscrete for single-antibiotic-per-patient)
   - Working examples of custom observation preprocessing

3. **Custom Environment Development**
   - Extension patterns (subclassing ABXAMREnv for multi-locale or multi-agent scenarios)
   - Wrapper composition (applying auxiliary rewards, observation filtering, etc.)
   - Integration requirements with RewardCalculator, PatientGenerator, AMRDynamicsBase
   - Validation checklist for custom environments

4. **Advanced Topics**
   - Crossresistance matrix configuration and impact on AMR coupling
   - Patient heterogeneity modeling and observation ordering
   - Multi-step episode structure and terminal condition handling
   - Gymnasium interface compliance details

5. **Debugging & Diagnostics**
   - Environment introspection methods
   - Common configuration errors and how to resolve them
   - Logging and TensorBoard integration for environment metrics
   - Visualization tools for environment trajectories

### Related Documents

For now, see:
- Tutorial 01: Basic Training (env instantiation examples)
- Tutorial 03: Custom Experiments (env configuration patterns)
- Tutorial 06: HRL Quick-Start (options in environment context)
- Tutorial 10: Advanced Heuristic Worker Subclassing (custom worker patterns)

### Next Steps

The user will provide guidance on:
1. Level of technical detail appropriate for target audience
2. Emphasis on common use cases vs. advanced extensibility patterns
3. Inclusion of visual diagrams or flowcharts
4. Examples drawn from existing experiments or template configurations
5. Cross-referencing strategy to other ABCs and concepts

---

**TODO**: Convert this placeholder to full documentation once user confirms scope and priorities.
