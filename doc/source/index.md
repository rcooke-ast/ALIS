# ALIS

ALIS (Absorption LIne Software) fits models to spectra. A model combines emission,
absorption and the instrument's resolution, and ALIS finds the values of its
parameters that best match the data, by minimising χ². You can prepare, run and
inspect a fit in ALIS's GUI, or run it from the command line with `run_alis`.

[Your first fit](user/start/first-fit.md)
: Install ALIS, then fit an example spectrum, in the GUI and from the command line.

[The GUI at a glance](user/gui/overview.md)
: The window, its tabs and how a project is saved.

[The model file](user/fitting/model-file.md)
: What a model file holds, for those who write their own.

Changing or extending ALIS? See the [Developer guide](developer/organisation.md).

% Sources: WF Overview; alis/__init__.py; the outline of
% claude_prompts/documentation_stage0.md.

```{toctree}
:caption: Getting started
:hidden:

user/start/introduction
user/start/installing
user/start/first-fit
```

```{toctree}
:caption: The GUI
:hidden:

user/gui/overview
user/gui/data
user/gui/regions
user/gui/components
user/gui/fit
user/gui/figure
user/gui/earlier-fits
```

```{toctree}
:caption: Fitting with ALIS
:hidden:

user/fitting/model-file
user/fitting/running
user/fitting/simulations
user/fitting/examples
user/fitting/troubleshooting
```

```{toctree}
:caption: Reference
:hidden:

user/reference/functions/index
user/reference/settings
user/reference/command-line
user/reference/atomic-data
user/reference/upgrading
```

```{toctree}
:caption: About
:hidden:

user/about
```

```{toctree}
:caption: Developer guide
:hidden:

developer/organisation
developer/model-functions
developer/gpu
developer/gui
developer/tests
developer/documentation
developer/contributing
developer/api
developer/changelog
```
