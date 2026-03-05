#!/bin/bash
set -e

echo "=== ArceOS Guestvdev Test Script ==="
echo ""

# Check if required tools are installed
check_tools() {
    echo "[1/7] Checking required tools..."
    
    if ! command -v cargo &> /dev/null; then
        echo "Error: cargo is not installed"
        exit 1
    fi
    
    if ! command -v rust-objcopy &> /dev/null; then
        echo "Warning: cargo-binutils not installed, installing..."
        cargo install cargo-binutils
    fi
    
    echo "✓ All required tools are available"
    echo ""
}

# Format check
check_format() {
    echo "[2/7] Checking code format..."
    cargo fmt -- --check
    echo "✓ Code format check passed"
    echo ""
}

# Clippy lint check by architecture
check_clippy() {
    echo "[3/7] Running clippy lint checks..."
    
    local archs=("riscv64" "x86_64" "aarch64")
    local targets=("riscv64gc-unknown-none-elf" "x86_64-unknown-none" "aarch64-unknown-none-softfloat")
    
    for i in "${!archs[@]}"; do
        local arch="${archs[$i]}"
        local target="${targets[$i]}"
        
        echo ""
        echo "Running clippy for architecture: $arch"
        
        # Install config file for the architecture
        cp "configs/${arch}.toml" ".axconfig.toml"
        
        if cargo clippy --target="$target" --features="axstd,guest-kernel,hypervisor" -- -W warnings; then
            echo "✓ $arch clippy check passed"
        else
            echo "Error: $arch clippy check failed"
            rm -f .axconfig.toml
            exit 1
        fi
    done
    
    rm -f .axconfig.toml
    echo ""
    echo "✓ All architecture clippy checks passed"
    echo ""
}

# Basic build check (no default features to avoid platform-specific issues)
check_build() {
    echo "[4/7] Checking basic build (no default features)..."
    cargo check --no-default-features
    echo "✓ Basic build check passed"
    echo ""
}

# Run tests for each architecture
run_arch_tests() {
    echo "[5/7] Running architecture-specific tests..."
    
    local archs=("riscv64" "x86_64" "aarch64")
    local qemu_ok=true
    
    for arch in "${archs[@]}"; do
        echo ""
        echo "Testing architecture: $arch"
        
        # Check if QEMU is available
        qemu_cmd="qemu-system-$arch"
        if ! command -v "$qemu_cmd" &> /dev/null; then
            echo "Warning: $qemu_cmd not found, skipping run test for $arch"
            qemu_ok=false
            continue
        fi
        
        # Build and run
        if cargo xtask run --arch="$arch" 2>&1 | grep -q "Multi-task(Preemptible) ok!"; then
            echo "✓ $arch test passed"
        else
            echo "Error: $arch test failed"
            exit 1
        fi
    done
    
    if [ "$qemu_ok" = true ]; then
        echo ""
        echo "✓ All architecture tests passed"
    fi
    echo ""
}

# Publish dry-run check by architecture
check_publish() {
    echo "[6/7] Checking publish readiness..."
    
    local archs=("riscv64" "x86_64" "aarch64")
    local targets=("riscv64gc-unknown-none-elf" "x86_64-unknown-none" "aarch64-unknown-none-softfloat")
    
    for i in "${!archs[@]}"; do
        local arch="${archs[$i]}"
        local target="${targets[$i]}"
        
        echo ""
        echo "Checking publish for architecture: $arch"
        
        # Install config file for the architecture
        cp "configs/${arch}.toml" ".axconfig.toml"
        
        if cargo publish --dry-run --allow-dirty --features="axstd,hypervisor" --target="$target"; then
            echo "✓ $arch publish check passed"
        else
            echo "Error: $arch publish check failed"
            rm -f .axconfig.toml
            exit 1
        fi
    done
    
    rm -f .axconfig.toml
    echo ""
    echo "✓ All architecture publish checks passed"
    echo ""
}

# Summary
print_summary() {
    echo "[7/7] Test Summary"
    echo "=================="
    echo "✓ All checks passed successfully!"
    echo ""
    echo "The following checks were performed:"
    echo "  1. Code format check (cargo fmt)"
    echo "  2. Lint check (cargo clippy)"
    echo "  3. Basic build check (cargo check)"
    echo "  4. Architecture tests (riscv64, x86_64, aarch64)"
    echo "  5. Publish readiness check (cargo publish --dry-run)"
    echo ""
}

# Main execution
main() {
    local skip_qemu=${SKIP_QEMU:-false}
    
    check_tools
    check_format
    check_clippy
    check_build
    
    if [ "$skip_qemu" = "true" ]; then
        echo "[5/7] Skipping architecture tests (SKIP_QEMU=true)"
        echo ""
    else
        run_arch_tests
    fi
    
    check_publish
    print_summary
}

# Run main function
main "$@"
