# QuasiSteadyStateTEG
Quasi Steady State 1D Thermoelectric model to evaluate the potential electrical output

This code describes the principles behind the concept for an innovative ThermoElectric Generator (TEG), which can adapt automatically to the supplied thermal power. The concept relies on the addition of a thermal buffer between the heat source and the TEG modules enabling control over the recovered thermal level irrespective of the thermal input variability. This allows the maximization of the use of the waste heat source without the risk of overheating or thermal dilution.

The engine conditions necessary to fulfil the driving cycle are predicted using a steady-state engine model map, from which the exhaust mass flow and temperature are extracted. The file WLTP.csv is an example of the thermal output of a driving cycle.

The model divides the longitudinal direction into multiple numerical sections, slices or nodes with one-dimensional heat transfer calculations being performed for each node. The calculations were performed under quasi steady state conditions. Although these are simplifications under the highly variable thermal load typical of driving cycles, they allow one to evaluate the merit of the concept, namely the order of magnitude of the heat transfer and electrical power output as well as some relevant limitations and optimization targets.
