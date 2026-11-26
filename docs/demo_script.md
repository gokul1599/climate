# MonsoonVerse Demonstration Script

1. **System Health & Architecture**: Show the live backend status badge (`MonsoonVerse: ready`) and CPU inference latency in ms.
2. **Normal 10-Day Forecast**: Select the Monsoon Season validation window and `Total Precipitation (tp)`. Scrub the 10-day slider from Day 1 to Day 10 to inspect spatial evolution across India.
3. **Warm Bay of Bengal Scenario**: Select `Warm Bay of Bengal` (+3.0 K `t2m`, +15% `tcwv` over 8°–20°N, 82°–94°E) and click **Run 10-Day Simulation**. Compare the Normal, Perturbed, and Difference (`Perturbed − Normal`) maps.
4. **Arabian Sea Monsoon Stall**: Switch to `Arabian Sea Monsoon Stall` (-40% `u10`/`v10` over 8°–20°N, 65°–75°E) and observe how wind-reduction alters moisture transport and downstream rainfall.
5. **Zero-Perturbation Check**: Run `Zero Perturbation Check` to confirm the difference map is identically zero (deterministic consistency).
6. **Impact Explorer**: Review the regional table for rainfall anomalies, heavy-rain cell-days, and the explicit scientific caveats.
