# Sources and attribution

## Simulator

The files under `vendor/virtual-fly-lab` are derived from [Leon-Av/virtual-fly-lab](https://github.com/Leon-Av/virtual-fly-lab), commit `0221edce5acbd625baa5f60b5797813a7577f384`.

The upstream MIT license is preserved in [vendor/virtual-fly-lab/LICENSE](vendor/virtual-fly-lab/LICENSE). Local changes include propagation direction, CPU/GPU parity, model parameters and graph preparation. Only the source files used by Droffel are included.

## Data and runtime

- [MaleCNS v1.0](https://storage.googleapis.com/flyem-male-cns/index.html): annotations, neurotransmitter predictions, directed connectivity and selected skeletons. Downloads are kept outside git and recorded with source URLs and SHA-256 hashes in `data/sources.lock.json` and morphology metadata.
- [Godot Engine](https://godotengine.org/): the pinned Windows runtime is downloaded from the official release. Godot is distributed under the MIT license; see the runtime's accompanying license and [Godot license page](https://godotengine.org/license/).
- [The Binding of Isaac: Repentance](https://store.steampowered.com/app/1426300/The_Binding_of_Isaac_Repentance/): required separately. No game executable, assets or proprietary scripts are included.

The project does not claim biological validation of game behaviour. Anatomical data, simplified neural dynamics and engineered game decisions are distinct parts of the system.
