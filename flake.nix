{
  description = "JAX CUDA via pip/uv inside Nix shell";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos-cuda.org"
      "https://cuda-maintainers.cachix.org"  # keep as fallback
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "cache.nixos-cuda.org-1:dn11R2MsKRK0LMjxoJFO0h5L3fK3TnpbcFMZAGYlCGE="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
        cuda = pkgs.cudaPackages_13;
        python = pkgs.python311;
        cudaLibs = with cuda; [
          cudatoolkit
          cudnn
          libcusparse_lt
          # nccl          
        ];
        fakeLocalConfig = pkgs.writeShellScript "ldconfig" ''
          if [ "$1" = "-p" ]; then
            echo "libcuda.so.1 (libc6,x86-64) => /run/opengl-driver/lib/libcuda.so.1"
            echo "libcuda.so (libc6,x86-64) => /run/opengl-driver/lib/libcuda.so.1"
          else
            exec /sbin/ldconfig "$@"
          fi
        '';
      in {
        devShells.default = pkgs.mkShell {

          packages = [
            python
            pkgs.uv
            ] ++ cudaLibs ++ [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.binutils
            pkgs.ffmpeg
            pkgs.graphviz
            pkgs.glibc.bin # this adds it to the path
            pkgs.glibc
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (cudaLibs ++ [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            # pkgs.glibc
          ]) + ":/run/opengl-driver/lib:${cuda.cudatoolkit}/lib";
          CUDA_PATH = cuda.cudatoolkit;
          XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cuda.cudatoolkit}";
          NIX_LDFLAGS = "-L/run/opengl-driver/lib -L${cuda.cudatoolkit}/lib";

          shellHook = ''

            mkdir -p /tmp/nix-ldconfig
            ln -sf ${fakeLocalConfig} /tmp/nix-ldconfig/ldconfig
            export PATH="/tmp/nix-ldconfig:$PATH"

            export PROJECT_ROOT="$(pwd)"
            # this allows imports to work even without installing the packages
            export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

            export UV_TORCH_BACKEND=cu130
            export UV_LINK_MODE=copy
            export UV_PYTHON=${python}/bin/python
            export UV_PROJECT_ENVIRONMENT="$(pwd)/.venv"
            export UV_PYTHON_DOWNLOAD=never

            # Create .venv with the Nix python if it doesn't exist yet
            if [ ! -d ".venv" ]; then
              uv venv .venv --python ${python}/bin/python
            fi

            source .venv/bin/activate
            uv sync

            export XLA_PYTHON_CLIENT_PREALLOCATE=false

            # some CUDA libraries (e.g. cuBLAS) are not properly linked by PyTorch, so we need to add them to LD_LIBRARY_PATH manually
            uv pip install nvidia-nvshmem-cu12 --python .venv/bin/python
            NVSHMEM_DIR=$(python -c "import importlib.util; spec = importlib.util.find_spec('nvidia.nvshmem'); print(spec.submodule_search_locations[0]) if spec else exit(1)" 2>/dev/null)
            echo "NVSHMEM_DIR: $NVSHMEM_DIR"
            if [ -n "$NVSHMEM_DIR" ]; then
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVSHMEM_DIR/lib"
            fi

            echo "------------------------------"
            echo "Entered Nix dev shell for system: ${system}"
            echo "Python path: $(which python)"
            echo "GLIBC version: $(ldd --version | head -n1)"
            echo "CUDA version: $(nvcc --version | grep release)"
            echo "PyTorch version: $(python -c 'import torch; print(torch.__version__, end=" "); print(torch.version.cuda)')"
            echo "------------------------------"
          '';
        };
      });
}
