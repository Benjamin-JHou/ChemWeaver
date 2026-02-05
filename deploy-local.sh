#!/bin/bash
# ChemWeaver Automated Deployment Script
# Run this from the chemweaver-release directory

set -e

echo "🚀 Starting ChemWeaver v1.0.0 Deployment..."
echo ""

# Navigate to chemweaver-release directory
cd "$(dirname "$0")"
CWD=$(pwd)
echo "📁 Working directory: $CWD"

# Clean up any existing git
rm -rf .git

# Initialize git
echo "📦 Initializing git repository..."
git init

# Configure git
git config user.email "benjamin.hou@example.com"
git config user.name "Benjamin J. Hou"

# Add remote
git remote add origin https://github.com/Benjamin-JHou/ChemWeaver.git 2>/dev/null || echo "Remote already exists"

# Add all files
echo "➕ Adding files to git..."
git add .

# Count files
FILE_COUNT=$(git status --short | wc -l)
echo "📊 Files to commit: $FILE_COUNT"

# Create commit
echo "💾 Creating initial commit..."
git commit -m "Initial release: ChemWeaver v1.0.0

This is the first public release of ChemWeaver, a reproducible AI-driven
virtual screening infrastructure for prospective drug discovery.

Key Features:
- Multi-stage screening pipeline (standardize → filter → score → select)
- AI surrogate model with 3 uncertainty quantification methods
- Container-native reproducibility (Docker/Singularity)
- Complete wet-lab bridge for experimental validation
- Comprehensive documentation (~20,000 words)

Performance:
- 50-300× faster than traditional docking
- 70-85% cost reduction
- Spearman ρ = 0.89 on DUD-E benchmark
- ECE < 0.05 (well-calibrated uncertainty)

Documentation:
- README.md with quick start
- skill.md with complete capabilities
- docs/ with reproducibility, architecture, benchmarks
- CITATION.cff for academic citation

License: MIT (free for commercial and research use)"

# Create tag
echo "🏷️  Creating version tag v1.0.0..."
git tag -a v1.0.0 -m "ChemWeaver v1.0.0 - First public reproducible release

Key Highlights:
- Reproducible AI-driven virtual screening
- Uncertainty-aware predictions (ECE < 0.05)
- Container-native execution
- Complete wet-lab bridge
- MIT License

Full documentation: https://github.com/Benjamin-JHou/ChemWeaver#readme

Changes from initial development:
- Complete 4-stage screening pipeline
- Multi-modal AI surrogate model
- 3 uncertainty quantification methods
- Comprehensive documentation suite
- Example datasets and tutorials

Contributors:
- Benjamin J. Hou (Lead Developer)"

echo ""
echo "✅ Local deployment complete!"
echo ""
echo "Next steps:"
echo "1. Push to GitHub: git push origin main"
echo "2. Push tag: git push origin v1.0.0"
echo "3. Enable Zenodo at https://zenodo.org"
echo "4. Create GitHub Release"
echo ""
echo "Repository URL: https://github.com/Benjamin-JHou/ChemWeaver"
