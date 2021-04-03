# basics
sudo apt-get update
sudo apt-get install -y gcc build-essential

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly
