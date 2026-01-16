{
  description = "Nix flake for this project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          cargo
          rustc
          rustfmt
          clippy
          rust-analyzer
          python314Packages.opencv4Full
          clang
          gtk2
        ];
        nativeBuildInputs = with pkgs; [
          pkg-configUpstream
          python314Packages.ninja
          python314Packages.cmake
        ];
        env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
        shellHook = ''
          export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

          echo "Building project..."
          cargo build
        '';
      };
    };
}
