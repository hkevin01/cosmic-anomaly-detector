#!/bin/bash

# Cosmic Anomaly Detector - Run Script
# This script sets up the environment and runs the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.9 or higher."
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 9 ]); then
        print_error "Python 3.9 or higher required. Found Python $PYTHON_VERSION"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION found"
}

# Ensure venv exists and activate (auto-create if missing)
ensure_venv() {
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating automatically..."
        setup_venv
        activate_venv
        if [ -f "requirements.txt" ]; then
            install_dependencies
        else
            print_warning "requirements.txt not found; skipping dependency install."
        fi
    else
        activate_venv
    fi
}

# Function to setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
            ensure_venv
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
            ensure_venv

# Function to activate virtual environment
activate_venv() {
    if [ -d "venv" ]; then
            ensure_venv
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Run with --setup first."
            ensure_venv
    fi
}

# Function to install dependencies
install_dependencies() {
            ensure_venv

    # Upgrade pip first
    pip install --upgrade pip

            ensure_venv
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed"
    else
            ensure_venv
        exit 1
    fi

    # Install development dependencies if available
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    fi

    # Install the package in development mode
    pip install -e .
    print_success "Package installed in development mode"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    pytest tests/ -v --cov=src/cosmic_anomaly_detector --cov-report=term-missing
    print_success "Tests completed"
}

# Function to run physics validation test
run_physics_test() {
    print_status "Running Phase 3 Physics Validation test..."
    export PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
    $PYTHON_CMD scripts/test_phase3.py
    print_success "Physics validation test completed"
}

# Function to run GUI
run_gui() {
    print_status "Starting Cosmic Anomaly Detector GUI..."
    export PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
    $PYTHON_CMD -m cosmic_anomaly_detector.gui.main_window
}

# Function to run CLI
run_cli() {
    print_status "Starting Cosmic Anomaly Detector CLI..."
    export PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
    $PYTHON_CMD -m cosmic_anomaly_detector.cli "$@"
}

# Function to run batch analysis
run_batch() {
    print_status "Starting batch analysis..."
    export PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
    $PYTHON_CMD scripts/batch_analyze.py "$@"
}

# Function to create sample data
create_sample_data() {
    print_status "Creating sample data..."
    mkdir -p data/samples
    export PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
    $PYTHON_CMD -c "
from tests.conftest import TestDataGenerator
import os

# Create sample JWST FITS files
generator = TestDataGenerator()

# Normal image
hdul_normal = generator.create_mock_jwst_fits(1024, 1024, 15, False)
hdul_normal.writeto('data/samples/jwst_normal.fits', overwrite=True)

# Image with anomaly
hdul_anomaly = generator.create_mock_jwst_fits(1024, 1024, 10, True)
hdul_anomaly.writeto('data/samples/jwst_with_anomaly.fits', overwrite=True)

print('Sample FITS files created in data/samples/')
"
    print_success "Sample data created in data/samples/"
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."

    # Check for required system packages
    MISSING_PACKAGES=""

    # Check for development headers (needed for some Python packages)
    if ! dpkg -l | grep -q python3-dev; then
        MISSING_PACKAGES="$MISSING_PACKAGES python3-dev"
    fi

    # Check for FITS libraries
    if ! ldconfig -p | grep -q libcfitsio; then
        MISSING_PACKAGES="$MISSING_PACKAGES libcfitsio-dev"
    fi

    # Check for HDF5 libraries
    if ! ldconfig -p | grep -q libhdf5; then
        MISSING_PACKAGES="$MISSING_PACKAGES libhdf5-dev"
    fi

    if [ -n "$MISSING_PACKAGES" ]; then
        print_warning "Missing system packages: $MISSING_PACKAGES"
        print_status "Install with: sudo apt-get install$MISSING_PACKAGES"
    fi

    print_success "System requirements check completed"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."

    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # Remove test artifacts
    rm -rf .pytest_cache
    rm -rf .coverage
    rm -rf htmlcov
    rm -rf build
    rm -rf dist
    rm -rf *.egg-info

    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Cosmic Anomaly Detector - Run Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup              Set up development environment"
    echo "  gui                Run the GUI application (default)"
    echo "  cli [args]         Run the CLI application"
    echo "  test               Run tests"
    echo "  physics-test       Run Phase 3 physics validation test"
    echo "  batch [args]       Run batch analysis"
    echo "  sample-data        Create sample JWST data"
    echo "  check              Check system requirements"
    echo "  clean              Clean up temporary files"
    echo "  install            Install dependencies only"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # First time setup"
    echo "  $0                          # Run GUI (default)"
    echo "  $0 gui                      # Run GUI explicitly"
    echo "  $0 cli --help               # Show CLI help"
    echo "  $0 test                     # Run tests"
    echo "  $0 batch data/*.fits        # Batch analyze FITS files"
    echo "  $0 sample-data              # Create sample data"
    echo ""
}

# Main script logic
main() {
    # Parse command line arguments
    COMMAND="${1:-gui}"  # Default to GUI if no command specified

    case "$COMMAND" in
        "setup")
            print_status "Setting up Cosmic Anomaly Detector development environment..."
            check_python_version
            check_requirements
            setup_venv
            activate_venv
            install_dependencies
            create_sample_data
            print_success "Setup completed! Run '$0 gui' to start the GUI."
            ;;
        "gui")
            check_python_version
            activate_venv
            run_gui
            ;;
        "cli")
            shift  # Remove 'cli' from arguments
            check_python_version
            activate_venv
            run_cli "$@"
            ;;
        "test")
            check_python_version
            activate_venv
            run_tests
            ;;
        "physics-test")
            check_python_version
            activate_venv
            run_physics_test
            ;;
        "batch")
            shift  # Remove 'batch' from arguments
            check_python_version
            activate_venv
            run_batch "$@"
            ;;
        "sample-data")
            check_python_version
            activate_venv
            create_sample_data
            ;;
        "check")
            check_python_version
            check_requirements
            ;;
        "clean")
            cleanup
            ;;
        "install")
            check_python_version
            activate_venv
            install_dependencies
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
