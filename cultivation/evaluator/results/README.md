# Results of the exact evaluator

Scripts read and write this folder through `../results/`, so its top level keeps the circuits next to their outputs.

| Where | What |
|---|---|
| `*.stim` | circuits: `nn_cat_d*` nearest-neighbor ring and ladder (`_unf`: growth reversed), `nn_full_*` complete protocols, `vcat_d*` verified cats, `ucat_d*` unverified cats, `fold_n*` GHZ folds (`n` is the number of data qubits), `line_d*` line checks, `gidney_d5.stim` the published d=5 circuit with T gates |
| `*.json` | outputs of the scripts of the same name: certificates, witnesses, leading terms (`_c4`, `_c5`), searches of weight five and six (`_k5`, `_k6`, `_proxy_k5`, `_proxy_k6`) and rates |
| `logs/` | text logs of long runs, such as `late_ilp_ladder_d9.log` and `late_ilp_vcat_d9.log`, the certificates at code distance nine |
| `didle/` | the circuits of Table IV, with idle noise on the data and with gathered measurements (`_w8d`), and their rates |
| `rp2/` | the RP^2 circuit at d=5 in all 256 branches |

`../README.md` gives the command behind each result.
