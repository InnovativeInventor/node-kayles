target="x86_64-apple-darwin"
# target="x86_64-unknown-linux-gnu"

echo "NB: If you're using linux, change the target in this script (release.sh)!"

# Cleanup
rm -rf /tmp/pgo-data/*

# Generate profiles
RUSTFLAGS="-C target-cpu=native -Cprofile-generate=/tmp/pgo-data --target=$target" cargo build --release
echo "Running sample invocations for profiling and compiling optimization . . ."
for n in {1..3}; do ./target/release/non-attacking-queens --size 8 && sleep 0.5; done

llvm_profdata_loc=$(fd --full-path ~/.rustup/toolchains ~/.rustup/toolchains/*nightly*/lib/rustlib/*/bin/ | rg "llvm-profdata")
$llvm_profdata_loc merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Compile
RUSTFLAGS="-C target-cpu=native -Cprofile-use=/tmp/pgo-data/merged.profdata --target=$target" cargo build --release

# Cleanup
rm -rf /tmp/pgo-data/*
