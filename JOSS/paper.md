---
title: 'OpenReservoirComputing: GPU-Accelerated Reservoir Computing in JAX'
tags:
  - Python
  - JAX
  - reservoir computing
  - echo state networks
  - time-series forecasting
  - chaotic systems
authors:
  - name: Jan P. Williams
    orcid: 0009-0005-4955-0411
    affiliation: 1
    corresponding: true
  - name: Dima Tretiak
    orcid: 0009-0005-0419-7554
    affiliation: 1
  - name: Steven L. Brunton
    orcid: 0000-0002-6565-5118
    affiliation: 1
  - name: J. Nathan Kutz
    orcid: 0000-0002-6004-2275
    affiliation: "2, 3, 4"
  - name: Krithika Manohar
    orcid: 0000-0002-1582-6767
    affiliation: 1
affiliations:
  - name: Department of Mechanical Engineering, University of Washington, USA
    index: 1
    ror: 00cvxb145
  - name: Department of Applied Mathematics, University of Washington, USA
    index: 2
  - name: Department of Electrical and Computer Engineering, University of Washington, USA
    index: 3
  - name: Autodesk Research, London, UK
    index: 4
date: "18 February 2026"
bibliography: paper.bib
---

# Summary

OpenReservoirComputing (ORC) is a Python library for reservoir computing (RC) built on JAX and Equinox. RC is a form of machine learning that functions by lifting a low-dimensional sequence or signal into a high-dimensional dynamical system and training a simple, linear readout layer from the high-dimensional dynamics back to a lower-dimensional quantity of interest. The most common application of RC is time-series forecasting, where the goal is to predict a signal's future evolution. RC has achieved state-of-the-art performance on this task, particularly when applied to chaotic dynamical systems. RC can also perform classification and control tasks. ORC provides modular components for custom RC models and built-in models for forecasting, classification, and control. By building on JAX and Equinox, ORC offers GPU acceleration, JIT compilation, and automatic vectorization. These capabilities make prototyping new models faster, enable larger reservoir architectures, and allow seamless integration with other deep learning models.

# Statement of Need

Time-series prediction, classification, and control are fundamental tasks across science and engineering, arising in applications from climate modeling and fluid dynamics to robotics and neuroscience. Deep learning approaches to these tasks typically require large datasets, long training times, and expensive tuning of optimization hyperparameters. RC offers a compelling alternative. Since only the readout layer is trained via a single ridge regression, RC models can be trained in a fraction of the time required by comparable recurrent neural networks, often with less data and fewer hyperparameters to tune [@lukosevicius2009reservoir]. This makes RC particularly attractive for rapid prototyping, real-time applications, and data-limited settings. However, realizing these benefits in practice requires software that is both efficient and adaptable.

ORC's built-in models provide an easy entry point for users new to the field. In particular, a new user can supply their own time-series data to instantiate, train, and forecast in three simple lines of code. 

```python
import orc
U_train = ...
esn = orc.forecaster.ESNForecaster(data_dim=3, res_dim=400)
esn, R = orc.forecaster.train_RCForecaster(esn, U_train)
U_pred = esn.forecast(fcast_len=1000, res_state=R[-1])
```

Built-in visualization tools make it easy to evaluate model performance. Varying the hyperparameters of built-in models lets users explore how RC performance depends on configuration choices. ORC's JAX foundation makes scaling to higher-dimensional parallel reservoir architectures equally simple.

Much RC research is aimed at designing performant reservoir architectures. ORC makes this easy through its use of abstract base classes for Embedding, Driver, and Readout layers. Users need only define forward pass logic to integrate a new reservoir topology or readout strategy, while reusing the rest of the framework. This modular approach also enables ablation studies on how different components affect RC performance.
Because of ORC's functional approach in JAX, built-in and user-created models provide end-to-end differentiability by default. This enables gradient-based optimization of input sequences for control problems. This also makes ORC well suited to integrate with deep learning models such as those presented in [@ozalp2023reconstruction; @ozalp2025real].

# State of the Field
| | **ORC** | **ReservoirPy** | **RC.jl** |
|---|---|---|---|
| Language | Python | Python | Julia |
| GPU | $\checkmark$ | $\checkmark$* | $\checkmark$ |
| Auto. Differentiable | $\checkmark$ | $\times$ | $\checkmark$ |
| Parallelizable | $\checkmark$ | $\times$ | $\checkmark$ |
| Vectorizable | $\checkmark$ | $\times$ | $\checkmark$ |
| Forecasting | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Classification | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Control | $\checkmark$ | $\times$ | $\times$ |
| Continuous Time | $\checkmark$ | $\times$ | $\times$ |

Table 1: Comparison of reservoir computing libraries across key features. ✓ indicates full support; ✗ indicates no support. \*ReservoirPy's GPU support is available via its JAX backend (v0.4.0+) but does not fully exploit JAX's functional programming model. *Parallelizable* denotes native support for parallel RC architectures as in [@pathak2018model] and *vectorizable* denotes native support for vectorization (e.g. `vmap`).

The most commonly used open-source library for reservoir computing is ReservoirPy [@trouvain2020reservoirpy]. Like ORC, ReservoirPy provides built-in architectures and an API for custom layers. ReservoirPy was initially built on NumPy and SciPy with the maintainers adding a JAX backend in v0.4.0. However, ORC differs from ReservoirPy in several important ways.

First, ORC was *designed* on top of JAX [@jax2018github] and Equinox [@kidger2021equinox], which provide a different programming model based on functional transformations. This enables native GPU/TPU acceleration, JIT compilation, and composable transformations (`jit`, `vmap`, `grad`) that cannot be retrofitted into a NumPy-based architecture. While the JAX backend of ReservoirPy does improve performance, the API cannot fully exploit JAX's capabilities. For example, ORC's autoregressive forecast loop uses `jax.lax.scan`, avoiding Python overhead; ReservoirPy's object-oriented design makes this impossible.
Second, ORC has a different built-in feature set. ORC supports continuous-time reservoir dynamics via Diffrax [@kidger2021on], allowing users to define reservoir equations as ordinary differential equations solved with adaptive-step integrators. ORC also supports novel architectures such as Taylor-expanded and GRU-based drivers alongside standard echo state networks. ReservoirPy does not provide these functionalities. 
Third, ORC models compose seamlessly with other deep learning models in Equinox, whereas ReservoirPy is suited to standalone RC tasks. 

Other open source libraries for RC include Pytorch-ESN [@nardo2018pytorchesn] and ReservoirComputing.jl [@martinuzzi2022reservoircomputing]. Pytorch-ESN allows for integration with other PyTorch models, but is not as widely adopted and offers a much more limited feature set than ORC or ReservoirPy. ReservoirComputing.jl is a widely used Julia library for RC that influenced many of ORC's design choices,  but lacks Python ML ecosystem integration. Table 1 summarizes the functionality of ORC, ReservoirPy, and ReservoirComputing.jl. 


# Software Design

![ORC three-layer pipeline architecture. Each reservoir computer (RC) is decomposed into (i) an ``embedding'' function that lifts a low-dimensional signal $u_t$ to the reservoir dimension, (ii) a ``driver'' function that propagates the reservoir state, and (iii) a ``readout'' that maps back to a target $y$. For control and forecasting RCs, the target $y$ is typically $u_{t+1}$ (either in the presence of a forcing term or not) and for classification the target $y$ is a label. \label{fig:architecture}](../imgs/architecture_paper-larger.pdf)

ORC models are decomposed into three components, illustrated in \autoref{fig:architecture}: (i) an embedding $f_E$ that lifts a low-dimensional input signal $u_t$ to a high-dimensional space, (ii) a driver $f_R$ that propagates the high-dimensional state $r_t$, and (iii) a readout $f_O$ that maps the latent state back to an approximation of some low-dimensional signal $y_t$. Depending on the task at hand, $y_t$ may be a future time-step of $u_t$, a label associated with input data, or some other target signal. ORC differs from many existing approaches that unify (i) and (ii). Separating the embedding from the reservoir state propagation allows for cleaner application of RC to non-standard tasks, such as acting as a surrogate model for model predictive control.
All components are implemented as Equinox modules [@kidger2021equinox], which are immutable pytree-registered objects. Model parameters (reservoir weights, readout matrices) are stored as JAX arrays within the module, and parameter updates produce new module instances via `eqx.tree_at` rather than mutating state in place. This functional design enables JAX's composable transformations to operate directly on model objects and allows ORC models to be composed with other Equinox modules.

ORC supports parallel reservoirs [@pathak2018model] by default via a `chunks` parameter in each module, enabling spatiotemporal RC methods unavailable in other libraries. This extra tensor dimension also allows for simple batching during training, avoiding excessive GPU VRAM requirements.

ORC provides unified training functions (`train_RCForecaster`, `train_RCClassifier`, `train_RCController`) that work with any model inheriting from the corresponding base class, including user-defined models with custom components. These functions delegate shape handling to the readout layer, allowing the same training function to handle both discrete and continuous-time models.

The library provides three built-in model classes: `ESNForecaster` for time-series prediction, `ESNClassifier` for sequence classification, and `ESNController` for learning control policies with exogenous control inputs. Each composes embedding, driver, and readout components and provides task-specific methods (`forecast`, `classify`, `apply_control`). Users who need custom architectures can subclass the abstract base classes, define only the components that differ, and immediately use the unified training functions without reimplementing teacher forcing, autoregressive prediction, or ridge regression.

ORC also includes a data generation module with ODE and PDE integrators for standard benchmark systems, including the Lorenz-63 attractor, Rössler system, double pendulum, Lorenz-96 model, and the Kuramoto-Sivashinsky equation, all implemented using Diffrax.

# Research Impact Statement

![ORC with GPU acceleration enables significantly faster performance than ReservoirPy, even when using ReservoirPy's JAX backend. Panel (a) shows the time per forecast step of RC models with varying reservoir dimension trained to forecast the Lorenz system, while panel (b) shows training time for RC models with fixed reservoir dimension of 2000 but varying number of training samples. Performance of the two libraries with and without GPU acceleration are shown. GPU results were obtained running on an NVIDIA A40 GPU and CPU results were obtained with an Apple M2 chip. \label{fig:comp}](../imgs/benchmark_results_combined.png) 

ORC enables easy reimplementation of architectures that integrate RC with larger neural networks [@ozalp2023reconstruction; @ozalp2025real], and makes it easier to iterate on these ideas than any existing library. We also benchmark ORC against ReservoirPy across reservoir sizes in \autoref{fig:comp} and number of training samples. We find that with GPU acceleration, ORC scales far more favorably than ReservoirPy. Moreover, ORC is the only current package that supports training RC for control tasks, thanks to its efficiency and the end-to-end differentiability that JAX and Equinox provide. RC-based control has recently been shown to be advantageous over other forms of RNN-based MPC [@williams2024reservoir]. ORC's performance also makes it well suited for the parallel architectures needed to extend RC to higher-dimensional settings.


# AI Usage Disclosure

Claude (Anthropic) was used for code assistance during code debugging, proofreading this paper, and the generation of the architecture visualization SVG. ChatGPT (OpenAI) was used to generate the ORC logo. All generated code was reviewed, tested, and validated by the authors.

# Acknowledgements
The authors acknowledge support from the National Science Foundation AI Institute in Dynamic Systems (grant number 2112085). The authors also thank Anastasia Bizyaeva, Noa Kaplan, Ling-Wei Kong for insightful conversations.
