{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
        overlays = [ (import rust-overlay) ];
      };
      # Use the pinned Nixpkgs defaults to match cache.nixos-cuda.org:
      # Python 3.14, PyTorch 2.13 and CUDA 12.9. Custom CUDA or Python
      # overrides change store paths and can trigger substantial rebuilds.
      python = pkgs.python3.withPackages (ps: with ps; [
        requests
        numpy
        torch
        torchvision
        peft
        transformers
        bitsandbytes
        datasets
        accelerate
      ]);
      rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ python ];
        nativeBuildInputs = [ rustToolchain pkgs.pkg-config ];
        buildInputs = [ pkgs.yarn pkgs.openssl ];

        shellHook = ''
          # Use the NVIDIA driver matching the running NixOS kernel.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export TRITON_PTXAS_PATH="${pkgs.cudaPackages.cuda_nvcc}/bin/ptxas"
          export RUST_SRC_PATH=${pkgs.rustPlatform.rustLibSrc}
        '';
      };
    };
}
