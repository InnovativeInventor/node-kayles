# basics
sudo apt-get update
sudo apt-get install -y gcc build-essential mingw-w64

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly
rustup component add llvm-tools-preview
bash setup.sh
