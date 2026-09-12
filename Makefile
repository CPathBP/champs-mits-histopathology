# Environment targets.
#
#   make env         create the analysis environment at ENV_PREFIX
#   make env-gpu     add the CUDA-compiled extensions (needs nvcc; MamMIL only)
#   make env-ssl     add the packages the in-domain encoder's training needs
#   make env-check   verify versions, dependency consistency, and, when
#                    DATA_ROOT is set, the primary-data checksums
#
# ENV_PREFIX defaults to ./.env; set it to place the environment elsewhere.
# Trident is installed unmodified from upstream at v0.2.3 and the DINOv2 fork
# at its tag, both without their declared dependencies (see env/EXTERNALS.md).

ENV_PREFIX ?= $(CURDIR)/.env
CONDA ?= conda
RUN = $(CONDA) run -p $(ENV_PREFIX) --no-capture-output
TRIDENT_SRC = git+https://github.com/mahmoodlab/TRIDENT.git@v0.2.3
DINOV2_SRC = git+https://github.com/CPathBP/dinov2.git@champs-blockexp-v1

.PHONY: env env-gpu env-ssl env-check

env:
	$(CONDA) env create -f env/environment.yml -p $(ENV_PREFIX)
	$(RUN) python -m pip install --no-deps -r env/requirements.lock
	$(RUN) python -m pip install --no-deps "trident @ $(TRIDENT_SRC)"
	$(RUN) python -m pip install --no-deps "dinov2 @ $(DINOV2_SRC)"
	@if [ -f pyproject.toml ]; then $(RUN) python -m pip install --no-deps -e . ; fi
	@echo "environment ready at $(ENV_PREFIX); run: make env-check"

env-gpu:
	$(RUN) python -m pip install --no-deps -r env/requirements-gpu-build.txt

env-ssl:
	$(RUN) python -m pip install --no-deps -r env/requirements-ssl.txt

env-check:
	$(RUN) python env/check.py
